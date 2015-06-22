
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
from abc import (
    ABCMeta,
    abstractmethod,
)
from contextlib import contextmanager

from bcolz import (
    carray,
    ctable,
)
from click import progressbar
from numpy import (
    array,
    full,
    uint32,
)
from pandas import DatetimeIndex
from six import (
    iteritems,
    string_types,
    with_metaclass,
)


from zipline.data.ffc.base import FFCLoader
from zipline.data.ffc.loaders._us_equity_pricing import (
    _compute_row_slices,
    _read_bcolz_data,
    load_adjustments_from_sqlite,
)

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)


US_EQUITY_PRICING_BCOLZ_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'day', 'id'
]
DAILY_US_EQUITY_PRICING_DEFAULT_FILENAME = 'daily_us_equity_pricing.bcolz'


@contextmanager
def passthrough(obj):
    yield obj


class BcolzDailyBarWriter(with_metaclass(ABCMeta)):
    """
    Class capable of writing daily OHLCV data to disk in a format that can be
    read efficiently by BcolzDailyOHLCVReader.

    Parameters
    ----------
    calendar : pandas.DatetimeIndex
        An index of dates to use for aligning the stored data.
    assets : pandas.Int64Index
        An index containing the asset_ids to be stored.

    See Also
    --------
    BcolzDailyBarReader : Consumer of the data written by this class.
    """

    def __init__(self, calendar, assets):
        self.calendar = calendar
        self.assets = assets

    @abstractmethod
    def gen_ctables(self, dates, assets):
        """
        Return an iterator of pairs of (asset_id, bcolz.ctable).
        """
        raise NotImplementedError()

    @abstractmethod
    def to_timestamp(self, raw_dt):
        """
        Convert a raw date entry produced by gen_ctables into a pandas
        Timestamp.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_uint32(self, array, colname):
        """
        Convert raw column values produced by gen_ctables into uint32 values.

        Parameters
        ----------
        array : np.array
            An array of raw values.
        colname : str, {'open', 'high', 'low', 'close', 'volume', 'day'}
            The name of the column being loaded.

        For output being read by the default BcolzOHLCVReader, data should be
        stored in the following manner:

        - Pricing columns (Open, High, Low, Close) should be stored as 1000 *
          as-traded dollar value.
        - Volume should be the as-traded volume.
        - Dates should be stored as seconds since midnight UTC, Jan 1, 1970.
        """
        raise NotImplementedError()

    def write(self, filename, show_progress=False):
        """
        Parameters
        ----------
        filename : str
            The location at which we should write our output.
        show_progress : bool
            Whether or not to show a progress bar while writing.

        Returns
        -------
        table : bcolz.ctable
            The newly-written table.
        """
        _iterator = self.gen_ctables(self.calendar, self.assets)
        if show_progress:
            pbar = progressbar(
                _iterator,
                length=len(self.assets),
                item_show_func=lambda i: i if i is None else str(i[0]),
                label="Merging asset files:",
            )
            with pbar as pbar_iterator:
                return self._write_internal(filename, pbar_iterator)
        return self._write_internal(filename, _iterator)

    def _write_internal(self, filename, iterator):
        """
        Internal implementation of write.

        `iterator` should be an iterator yielding pairs of (asset, ctable).
        """
        dates = self.calendar
        total_rows = 0
        first_row = {}
        last_row = {}
        calendar_offset = {}

        # Maps column name -> output carray.
        columns = {
            k: carray(array([], dtype=uint32))
            for k in US_EQUITY_PRICING_BCOLZ_COLUMNS
        }

        for asset_id, table in iterator:
            nrows = len(table)
            for column_name in columns:
                if column_name == 'id':
                    # We know what the content of this column is, so don't
                    # bother reading it.
                    columns['id'].append(full((nrows,), asset_id))
                    continue
                columns[column_name].append(
                    self.to_uint32(table[column_name][:], column_name)
                )

            # Bcolz doesn't support ints as keys in `attrs`, so convert
            # assets to strings for use as attr keys.
            asset_key = str(asset_id)

            # Calculate the index into the array of the first and last row
            # for this asset. This allows us to efficiently load single
            # assets when querying the data back out of the table.
            first_row[asset_key] = total_rows
            last_row[asset_key] = total_rows + nrows - 1
            total_rows += nrows

            # Calculate the number of trading days between the first date
            # in the stored data and the first date of **this** asset. This
            # offset used for output alignment by the reader.
            calendar_offset[asset_key] = self.calendar.get_loc(
                self.to_timestamp(table['day'][0])
            )

        # This writes the table to disk.
        full_table = ctable(
            columns=[
                columns[colname]
                for colname in US_EQUITY_PRICING_BCOLZ_COLUMNS
            ],
            names=US_EQUITY_PRICING_BCOLZ_COLUMNS,
            rootdir=filename,
            mode='w',
        )
        full_table.attrs['first_row'] = first_row
        full_table.attrs['last_row'] = last_row
        full_table.attrs['calendar_offset'] = calendar_offset
        full_table.attrs['calendar'] = dates.asi8.tolist()
        return full_table


class BcolzDailyBarReader(object):
    """
    Reader for raw pricing data written by BcolzDailyOHLCVWriter.

    A Bcolz CTable is comprised of Columns and Attributes.

    Columns
    -------
    The table with which this loader interacts contains the following columns:

    ['open', 'high', 'low', 'close', 'volume', 'day', 'id'].

    The data in these columns is interpreted as follows:

    - Price columns ('open', 'high', 'low', 'close') are interpreted as 1000 *
      as-traded dollar value.
    - Volume is interpreted as as-traded volume.
    - Day is interpreted as seconds since midnight UTC, Jan 1, 1970.
    - Id is the asset id of the row.

    The data in each column is grouped by asset and then sorted by day within
    each asset block.

    The table is built to represent a long time range of data, e.g. ten years
    of equity data, so the lengths of each asset block is not equal to each
    other. The blocks are clipped to the known start and end date of each asset
    to cut down on the number of empty values that would need to be included to
    make a regular/cubic dataset.

    When read across the open, high, low, close, and volume with the same
    index should represent the same asset and day.

    Attributes
    ----------
    The table with which this loader interacts contains the following
    attributes:

    first_row : dict
        Map from asset_id -> index of first row in the dataset with that id.
    last_row : dict
        Map from asset_id -> index of last row in the dataset with that id.
    calendar_offset : dict
        Map from asset_id -> calendar index of first row.
    calendar : list[int64]
        Calendar used to compute offsets, in asi8 format (ns since EPOCH).

    We use first_row and last_row together to quickly find ranges of rows to
    load when reading an asset's data into memory.

    We use calendar_offset and calendar to orient loaded blocks within a
    range of queried dates.
    """
    def __init__(self, table):
        if isinstance(table, string_types):
            table = ctable(rootdir=table, mode='r')

        self._table = table
        self._calendar = DatetimeIndex(table.attrs['calendar'], tz='UTC')
        self._first_rows = {
            int(asset_id): start_index
            for asset_id, start_index in iteritems(table.attrs['first_row'])
        }
        self._last_rows = {
            int(asset_id): end_index
            for asset_id, end_index in iteritems(table.attrs['last_row'])
        }
        self._calendar_offsets = {
            int(id_): offset
            for id_, offset in iteritems(table.attrs['calendar_offset'])
        }

    def compute_row_slices(self, dates, assets):
        """
        Parameters
        ----------

        Returns
        -------

        See Also
        --------
        """
        query_start = self._calendar.get_loc(dates[0])
        query_stop = self._calendar.get_loc(dates[-1])

        # Sanity check that the requested date range matches our calendar.
        # This could be removed in the future if it's materially affecting
        # performance.
        query_dates = self._calendar[query_start:query_stop + 1]
        if not (query_dates.values == dates.values).all():
            raise ValueError("Incompatible calendars!")

        return _compute_row_slices(
            self._first_rows,
            self._last_rows,
            self._calendar_offsets,
            query_start,
            query_stop,
            assets,
        )

    def load_raw_arrays(self, columns, dates, assets):
        first_rows, last_rows, offsets = self.compute_row_slices(dates, assets)
        return _read_bcolz_data(
            self._table,
            (len(dates), len(assets)),
            [column.name for column in columns],
            first_rows,
            last_rows,
            offsets,
        )


class SQLiteAdjustmentLoader(object):
    """
    Loads adjustments based on corporate actions from a SQLite database.

    The database has tables for mergers, dividends, and splits.

    Each table has the columns:
    - sid, the asset identifier

    - effective_date, the midnight of date, in seconds, on which the adjustment
    starts. Adjustments are applied on the effective_date and all dates before
    it.

    - ratio, the price and/or volume multiplier.

    Corporate Actions Types:

    mergers, modify the price (ohlc)

    splits, modify the price (ohlc), and the volume. Volume is modify the
    inverse of the price adjustment.

    dividends, modify the price (ohlc). The dividend ratio is calculated as:
    1.0 - (dividend / "close of the market day before the ex_date"
    """

    def __init__(self, conn):
        self.conn = conn

    def load_adjustments(self, columns, dates, assets):
        return load_adjustments_from_sqlite(self.conn, columns, dates, assets)


class USEquityPricingLoader(FFCLoader):

    def __init__(self, raw_price_loader, adjustments_loader):
        self.raw_price_loader = raw_price_loader
        self.adjustments_loader = adjustments_loader

    def load_adjusted_array(self, columns, dates, assets):
        raw_arrays = self.raw_price_loader.load_raw_arrays(
            columns,
            dates,
            assets,
        )
        adjustments = self.adjustments_loader.load_adjustments(
            columns,
            dates,
            assets,
        )

        return [
            adjusted_array(raw_array, NOMASK, col_adjustments)
            for raw_array, col_adjustments in zip(raw_arrays, adjustments)
        ]
