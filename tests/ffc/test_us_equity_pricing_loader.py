#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for zipline.data.ffc.loaders.us_equity_pricing
"""
import os
import sqlite3
from unittest import TestCase

from bcolz import ctable
from nose_parameterized import parameterized
import numpy as np
from numpy import (
    arange,
    datetime64,
    float64,
    full,
    iinfo,
    uint32,
)
from numpy.testing import assert_array_equal
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Timedelta,
    Timestamp,
)
from pandas.util.testing import assert_index_equal
from testfixtures import TempDirectory

from zipline.data.adjustment import Float64Multiply
from zipline.data.equities import USEquityPricing
from zipline.data.ffc.loaders.us_equity_pricing import (
    BcolzDailyBarReader,
    BcolzDailyBarWriter,
    SQLiteAdjustmentLoader,
)
from zipline.finance.trading import TradingEnvironment

UINT_32_MAX = iinfo(uint32).max

# Test calendar ranges over the month of June 2015
#      June 2015
# Mo Tu We Th Fr Sa Su
#  1  2  3  4  5  6  7
#  8  9 10 11 12 13 14
# 15 16 17 18 19 20 21
# 22 23 24 25 26 27 28
# 29 30
TEST_CALENDAR_START = Timestamp('2015-06-01', tz='UTC')
TEST_CALENDAR_STOP = Timestamp('2015-06-30', tz='UTC')

TEST_QUERY_START = Timestamp('2015-06-10', tz='UTC')
TEST_QUERY_STOP = Timestamp('2015-06-20', tz='UTC')
TEST_QUERY_COLUMNS = [USEquityPricing.close, USEquityPricing.volume]

# One asset for each of the cases enumerated in load_raw_arrays_from_bcolz.
EQUITY_INFO = DataFrame(
    [
        # 1) The equity's trades start and end before query.
        {'start_date': '2015-06-01', 'end_date': '2015-06-05'},
        # 2) The equity's trades start and end after query.
        {'start_date': '2015-06-22', 'end_date': '2015-06-30'},
        # 3) The equity's data covers all dates in range.
        {'start_date': '2015-06-02', 'end_date': '2015-06-30'},
        # 4) The equity's trades start before the query start, but stop
        #    before the query end.
        {'start_date': '2015-06-01', 'end_date': '2015-06-15'},
        # 5) The equity's trades start and end during the query.
        {'start_date': '2015-06-12', 'end_date': '2015-06-18'},
        # 6) The equity's trades start during the query, but extend through
        #    the whole query.
        {'start_date': '2015-06-15', 'end_date': '2015-06-25'},
    ],
    index=arange(1, 7),
    columns=['start_date', 'end_date'],
).astype(datetime64)

TEST_QUERY_ASSETS = EQUITY_INFO.index
EPOCH = pd.Timestamp(0, tz='UTC')


def nanos_to_seconds(nanos):
    return nanos / (1000 * 1000 * 1000)


def strings_to_seconds(str_series):
    return nanos_to_seconds(
        DatetimeIndex(str_series.values, tz='UTC').asi8
    )


def seconds_to_timestamp(seconds):
    return Timestamp(0, tz='UTC') + Timedelta(seconds=seconds)


class DailyBarTestWriter(BcolzDailyBarWriter):
    """
    Bcolz writer that creates synthetic data based on asset lifetime metadata.

    For a given asset/date/column combination, we generate a corresponding raw
    value using the formula:

    data(asset, date, column) = (100,000 * asset_id)
                              + (10,000 * column_num)
                              + (date - Jan 1 2000).days  # ~6000 for 2015
    where:
        column_num('open') = 0
        column_num('high') = 1
        column_num('low') = 2
        column_num('close') = 3
        column_num('volume') = 4

    We use days since Jan 1, 2000 to guarantee that there are no collisions
    while also the produced values smaller than UINT32_MAX / 1000.

    Parameters
    ----------
    asset_info : DataFrame
        DataFrame with asset_id as index and 'start_date'/'end_date' columns.
    calendar : DatetimeIndex
        Dates to use to compute offsets.
    """
    OHLCV = ('open', 'high', 'low', 'close', 'volume')
    PSEUDO_EPOCH = Timestamp('2000-01-01', tz='UTC')

    def __init__(self, asset_info, calendar):
        super(DailyBarTestWriter, self).__init__(calendar, asset_info.index)
        self._asset_info = asset_info
        self._frames = {}
        for asset_id in asset_info.index:
            start, end = asset_info.ix[asset_id, ['start_date', 'end_date']]
            asset_dates = calendar[
                calendar.get_loc(start):calendar.get_loc(end) + 1
            ]

            opens, highs, lows, closes, volumes = self._make_raw_data(
                asset_id,
                asset_dates,
            )
            days = asset_dates.asi8
            ids = full((len(asset_dates),), asset_id, dtype=uint32)
            self._frames[asset_id] = DataFrame(
                {
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes,
                    'id': ids,
                    'day': days,
                },
            )

    def _make_raw_data(self, asset_id, asset_dates):
        """
        Generate 'raw' data that encodes information about the asset.

        See class docstring for a description of the data format.
        """
        assert asset_dates[0] > self.PSEUDO_EPOCH
        OHLCV_COLUMN_COUNT = len(self.OHLCV)
        data = full(
            (len(asset_dates), OHLCV_COLUMN_COUNT),
            asset_id * (100 * 1000),
            dtype=uint32,
        )

        # Add 10,000 * column-index to each column.
        data += arange(OHLCV_COLUMN_COUNT) * (10 * 1000)

        # Add days since Jan 1 2001 for each row.
        data += (asset_dates - self.PSEUDO_EPOCH).days[:, None]

        # Return data as iterable of column arrays.
        return (data[:, i] for i in range(OHLCV_COLUMN_COUNT))

    @classmethod
    def expected_value(cls, asset_id, date, colname):
        """
        Check that the raw value for an asset/date/column triple is as
        expected.

        Used by tests to verify data written by a writer.
        """
        from_asset = asset_id * 100 * 1000
        from_colname = cls.OHLCV.index(colname) * (10 * 1000)
        from_date = (date - cls.PSEUDO_EPOCH).days
        return from_asset + from_colname + from_date

    # BEGIN SUPERCLASS INTERFACE
    def gen_ctables(self, dates, assets):
        for asset in assets:
            # Clamp stored data to the requested date range.
            frame = self._frames[asset].reset_index()
            yield asset, ctable.fromdataframe(frame)

    def to_timestamp(self, raw_dt):
        return Timestamp(raw_dt)

    def to_uint32(self, array, colname):
        if colname == 'day':
            return nanos_to_seconds(array)
        elif colname in {'open', 'high', 'low', 'close'}:
            # Data is stored as 1000 * raw value.
            assert array.max() < (UINT_32_MAX / 1000), "Test data overflow!"
            return array * 1000
        else:
            assert colname == 'volume', "Unknown column: %s" % colname
            return array
    # END SUPERCLASS INTERFACE


class DailyBarReaderWriterTestCase(TestCase):

    def setUp(self):
        all_trading_days = TradingEnvironment.instance().trading_days
        self.trading_days = all_trading_days[
            all_trading_days.get_loc(TEST_CALENDAR_START):
            all_trading_days.get_loc(TEST_CALENDAR_STOP) + 1
        ]

        self.asset_info = EQUITY_INFO
        self.writer = DailyBarTestWriter(
            self.asset_info,
            self.trading_days,
        )
        self.dir_ = TempDirectory()
        self.dir_.create()
        self.dest = self.dir_.getpath('daily_equity_pricing.bcolz')

    def tearDown(self):
        self.dir_.cleanup()

    @property
    def assets(self):
        return self.asset_info.index

    def trading_days_between(self, start, end):
        return self.trading_days[self.trading_days.slice_indexer(start, end)]

    def asset_start(self, asset_id):
        return self.asset_info.loc[asset_id]['start_date'].tz_localize('UTC')

    def asset_end(self, asset_id):
        return self.asset_info.loc[asset_id]['end_date'].tz_localize('UTC')

    def dates_for_asset(self, asset_id):
        start, end = self.asset_start(asset_id), self.asset_end(asset_id)
        return self.trading_days_between(start, end)

    def test_write_ohlcv_content(self):
        result = self.writer.write(self.dest)
        for column in DailyBarTestWriter.OHLCV:
            idx = 0
            data = result[column][:]
            multiplier = 1 if column == 'volume' else 1000
            for asset_id in self.assets:
                for date in self.dates_for_asset(asset_id):
                    self.assertEqual(
                        DailyBarTestWriter.expected_value(
                            asset_id,
                            date,
                            column
                        ) * multiplier,
                        data[idx],
                    )
                    idx += 1
            self.assertEqual(idx, len(data))

    def test_write_day_and_id(self):
        result = self.writer.write(self.dest)
        idx = 0
        ids = result['id']
        days = result['day']
        for asset_id in self.assets:
            for date in self.dates_for_asset(asset_id):
                self.assertEqual(ids[idx], asset_id)
                self.assertEqual(date, seconds_to_timestamp(days[idx]))
                idx += 1

    def test_write_attrs(self):
        result = self.writer.write(self.dest)
        expected_first_row = {
            '1': 0,
            '2': 5,   # Asset 1 has 5 trading days.
            '3': 12,  # Asset 2 has 7 trading days.
            '4': 33,  # Asset 3 has 21 trading days.
            '5': 44,  # Asset 4 has 11 trading days.
            '6': 49,  # Asset 5 has 5 trading days.
        }
        expected_last_row = {
            '1': 4,
            '2': 11,
            '3': 32,
            '4': 43,
            '5': 48,
            '6': 57,    # Asset 6 has 9 trading days.
        }
        expected_calendar_offset = {
            '1': 0,   # Starts on 6-01, 1st trading day of month.
            '2': 15,  # Starts on 6-22, 16th trading day of month.
            '3': 1,   # Starts on 6-02, 2nd trading day of month.
            '4': 0,   # Starts on 6-01, 1st trading day of month.
            '5': 9,   # Starts on 6-12, 10th trading day of month.
            '6': 10,  # Starts on 6-15, 11th trading day of month.
        }
        self.assertEqual(result.attrs['first_row'], expected_first_row)
        self.assertEqual(result.attrs['last_row'], expected_last_row)
        self.assertEqual(
            result.attrs['calendar_offset'],
            expected_calendar_offset,
        )
        assert_index_equal(
            self.trading_days,
            DatetimeIndex(result.attrs['calendar'], tz='UTC'),
        )

    def expected_read_values(self, dates, assets, column):
        if column == 'volume':
            dtype = uint32
            missing = 0
        else:
            dtype = float64
            missing = float('nan')

        data = full((len(dates), len(assets)), missing, dtype=dtype)
        for j, asset in enumerate(assets):
            start = self.asset_start(asset)
            stop = self.asset_end(asset)
            for i, date in enumerate(dates):
                # No value expected for dates outside the asset's start/end
                # date.
                if not (start <= date <= stop):
                    continue
                data[i, j] = DailyBarTestWriter.expected_value(
                    asset,
                    date,
                    column,
                )
        return data

    def _check_read_results(self, columns, assets, start_date, end_date):
        reader = BcolzDailyBarReader(self.writer.write(self.dest))
        dates = self.trading_days_between(start_date, end_date)
        results = reader.load_raw_arrays(columns, dates, assets)
        for column, result in zip(columns, results):
            assert_array_equal(
                result,
                self.expected_read_values(dates, assets, column.name),
            )

    @parameterized.expand([
        ([USEquityPricing.open],),
        ([USEquityPricing.close, USEquityPricing.volume],),
        ([USEquityPricing.volume, USEquityPricing.high, USEquityPricing.low],),
        (USEquityPricing.columns,),
    ])
    def test_read(self, columns):
        self._check_read_results(
            columns,
            self.assets,
            TEST_QUERY_START,
            TEST_QUERY_STOP,
        )

    def test_start_on_asset_start(self):
        """
        Test loading with queries that starts on the first day of each asset's
        lifetime.
        """
        columns = [USEquityPricing.high, USEquityPricing.volume]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_start(asset),
                end_date=self.trading_days[-1],
            )

    def test_start_on_asset_end(self):
        """
        Test loading with queries that start on the last day of each asset's
        lifetime.
        """
        columns = [USEquityPricing.close, USEquityPricing.volume]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_end(asset),
                end_date=self.trading_days[-1],
            )

    def test_end_on_asset_start(self):
        """
        Test loading with queries that end on the first day of each asset's
        lifetime.
        """
        columns = [USEquityPricing.close, USEquityPricing.volume]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.trading_days[0],
                end_date=self.asset_start(asset),
            )

    def test_end_on_asset_end(self):
        """
        Test loading with queries that end on the last day of each asset's
        lifetime.
        """
        columns = [USEquityPricing.close, USEquityPricing.volume]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.trading_days[0],
                end_date=self.asset_end(asset),
            )


# ADJUSTMENTS use the following scheme to indicate information about the value
# upon inspection.
#
# 1s place is the equity
#
# 0.1s place is the action type, with:
#
# splits, 1
# mergers, 2
# dividends, 3
#
# 0.001s is the date

SPLITS = [
    # Before query range, should be excluded.
    {'effective_date': '2015-06-03',
     'ratio': 1.103,
     'sid': 1},
    # First day of query range, should have last_row of 0
    {'effective_date': '2015-06-04',
     'ratio': 1.104,
     'sid': 1},
    # Second day of query range, should have last_row of 2
    {'effective_date': '2015-06-06',
     'ratio': 1.106,
     'sid': 1},
    # After query range, should be excluded.
    {'effective_date': '2015-06-09',
     'ratio': 1.109,
     'sid': 1},
    # Another action in query range, should have last_row of 1
    {'effective_date': '2015-06-05',
     'ratio': 2.105,
     'sid': 2},
]


MERGERS = [
    # Before query range, should be excluded.
    {'effective_date': '2015-06-03',
     'ratio': 1.203,
     'sid': 1},
    # First day of query range, should have last_row of 0
    {'effective_date': '2015-06-04',
     'ratio': 1.204,
     'sid': 1},
    # Second day of query range, should have last_row of 2
    {'effective_date': '2015-06-06',
     'ratio': 1.206,
     'sid': 1},
    # After query range, should be excluded.
    {'effective_date': '2015-06-09',
     'ratio': 1.209,
     'sid': 1},
    # Another action in query range, should have last_row of 2
    {'effective_date': '2015-06-06',
     'ratio': 2.206,
     'sid': 2},
]


DIVIDENDS = [
    # Before query range, should be excluded.
    {'effective_date': '2015-06-03',
     'ratio': 1.303,
     'sid': 1},
    # First day of query range, should have last_row of 0
    {'effective_date': '2015-06-04',
     'ratio': 1.304,
     'sid': 1},
    # Second day of query range, should have last_row of 2
    {'effective_date': '2015-06-06',
     'ratio': 1.306,
     'sid': 1},
    # After query range, should be excluded.
    {'effective_date': '2015-06-09',
     'ratio': 1.309,
     'sid': 1},
    # Another action in query range, should have last_row of 3
    {'effective_date': '2015-06-07',
     'ratio': 2.307,
     'sid': 2},
]


def create_adjustments_data(test_data_dir):
    db_path = os.path.join(test_data_dir, 'adjustments.db')
    conn = sqlite3.connect(db_path)

    splits_df = pd.DataFrame(SPLITS)
    splits_df['effective_date'] = strings_to_seconds(
        splits_df['effective_date'])

    mergers_df = pd.DataFrame(MERGERS)
    mergers_df['effective_date'] = strings_to_seconds(
        mergers_df['effective_date'])

    dividends_df = pd.DataFrame(DIVIDENDS)
    dividends_df['effective_date'] = strings_to_seconds(
        dividends_df['effective_date'])

    splits_df.to_sql('splits', conn)
    mergers_df.to_sql('mergers', conn)
    dividends_df.to_sql('dividends', conn)

    c = conn.cursor()
    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='dividends_sid', tn='dividends', cn='sid'))
    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='dividends_pay_date', tn='dividends', cn='effective_date'))

    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='mergers_sid', tn='mergers', cn='sid'))
    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='mergers_effective_date', tn='mergers', cn='effective_date'))

    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='splits_sid', tn='splits', cn='sid'))
    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='splits_effective_date', tn='mergers', cn='effective_date'))

    conn.close()

    return db_path


class UsEquityPricingLoaderTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = TempDirectory()
        cls.adjustments_test_data_path = create_adjustments_data(
            cls.test_data_dir.path
        )

    @classmethod
    def tearDownClass(cls):
        cls.test_data_dir.cleanup()

    def setUp(self):
        self.assets = TEST_QUERY_ASSETS
        all_trading_days = TradingEnvironment.instance().trading_days
        self.trading_days = all_trading_days[
            all_trading_days.slice_indexer(TEST_QUERY_START, TEST_QUERY_STOP)
        ]

    def test_load_adjustments_from_sqlite(self):
        conn = sqlite3.connect(self.adjustments_test_data_path)

        adjustments_loader = SQLiteAdjustmentLoader(conn)

        import nose.tools; nose.tools.set_trace()
        adjustments = adjustments_loader.load_adjustments(
            TEST_QUERY_COLUMNS,
            self.trading_days,
            self.assets,
        )

        close_adjustments = adjustments[0]
        volume_adjustments = adjustments[1]

        # See SPLITS, MERGERS, DIVIDENDS module variables for details of
        # expected values.
        EXPECTED_CLOSES = {
            # 2015-06-04
            0: [
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.104),
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.204),
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.304)
            ],
            1: [
                Float64Multiply(
                    first_row=0, last_row=1, col=1, value=2.105)
            ],
            2: [
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.106),
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.206),
                Float64Multiply(
                    first_row=0, last_row=2, col=1, value=2.206),
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.306)
            ],
            3: [
                Float64Multiply(
                    first_row=0, last_row=3, col=1, value=2.307000)
            ]
        }

        EXPECTED_VOLUMES = {
            0: [
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.0 / 1.104)
            ],
            1: [
                Float64Multiply(
                    first_row=0, last_row=1, col=1, value=1.0 / 2.105)
            ],
            2: [
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.0 / 1.106)
            ]}

        import nose.tools; nose.tools.set_trace()

        self.assertEqual(close_adjustments, EXPECTED_CLOSES)
        self.assertEqual(volume_adjustments, EXPECTED_VOLUMES)
