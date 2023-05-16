
import argparse
import xarray as xr
import numpy as np
from tqdm import tqdm
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import threading
import matplotlib
matplotlib.use("TkAgg")

class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.parse_args(open(values).read().split(), namespace)

def nckrig():
    global ds, rmse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--data-name', type=str, default='train.nc',
                                help="netCDF file (climate dataset) for training")
    parser.add_argument("-m", '--mask-name', type=str, default=None,
                                help="netCDF file (mask dataset). If None, it extracts the masks from the climate dataset")
    parser.add_argument("-t", '--data-type', type=str, default='tas', help="Variable type")
    parser.add_argument("--model", type=str, default="gaussian", help="Model for the variogram")
    parser.add_argument("-r", "--range", type=float, default=None, help="Range (in %) for parameter search")
    parser.add_argument("-s", "--nsearch", type=int, default=None, help="Number of points for parameter search for parameter search")
    parser.add_argument("-p", "--params", type=str, default=None, help="x0 params")
    parser.add_argument("-b", "--nbins", type=str, default="6", help="Number of bins")
    parser.add_argument("-v", "--varplot", action='store_true', help="Plot the variogram")
    parser.add_argument("-u", "--universal", action='store_true', help="Use universal kriging")
    parser.add_argument("-n", "--n-threads", type=int, default=1, help="Number of threads")
    parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile, help="Load all the arguments from a text file")
    parser.add_argument('--data-dir', type=str, default='',
                                help="Directory containing the climate datasets")
    parser.add_argument('--mask-dir', type=str, default='', help="Directory containing the mask datasets")
    parser.add_argument('--output-dir', type=str, default='',
                                help="Directory where the output files will be stored")
    args = parser.parse_args()

    model = args.model

    if args.universal:
        kriging = UniversalKriging
    else:
        kriging = OrdinaryKriging

    ds = xr.open_dataset(args.data_dir + args.data_name)

    if args.mask_name is None:
        mask = None
    else:
        mask = (1 - xr.open_dataset(args.mask_dir + args.mask_name)[args.data_type].values).astype(bool)
        assert ds[args.data_type].shape == mask.shape

    nbins = [int(i) for i in args.nbins.split(",")]

    if args.params is None:
        params = None
    else:
        params = [float(i) for i in args.params.split(",")]

    if args.range is not None and args.nsearch is not None:
        k = -1
        for x0 in params:
            k += 1
            d0 = args.range * x0 / 100
            params[k] = np.linspace(x0 - d0, x0 + d0, num=args.nsearch)
        params = [list(i) for i in np.array(np.meshgrid(*params)).T.reshape(-1, len(params))]
    else:
        params = [params]

    ntime = len(ds.time)
    t_chunks = np.array_split(np.arange(ntime), -(ntime // -args.n_threads))

    nconf = len(params)
    ntot = nconf * len(nbins)

    print("* Total number of configurations:", ntot)

    search = False
    varplot = args.varplot
    if ntot > 1:
        search = True
        varplot = False

    pbar = tqdm(total=ntot * len(t_chunks))

    dims = {}
    for dim in ds.dims:
        if "lon" in dim:
            dims["lon"] = dim
        if "lat" in dim:
            dims["lat"] = dim

    grid_lon = ds[dims["lon"]].values
    grid_lat = ds[dims["lat"]].values

    rmin = np.inf

    def reconstruct(t, selvar, grid_lat, grid_lon, mask, model, params, nbin, varplot, search):
        global ds, rmse

        vals = ds[selvar].values[t].copy()
        if mask is not None:
            vals[mask[t]] = np.nan
        idx = np.where(~np.isnan(vals))
        vals = vals[idx]
        lat = grid_lat[idx[0]]
        lon = grid_lon[idx[1]]

        OUK = kriging(np.array(lon), np.array(lat), np.array(vals), variogram_model=model, verbose=False,
                             variogram_parameters=params, nlags=nbin, enable_plotting=varplot)
        interp, ss1 = OUK.execute('grid', grid_lon, grid_lat)

        rmse[t] = np.sqrt(np.mean((ds[args.data_type][t].values - interp)**2))

        if not search:
            ds[args.data_type][t] = np.array(interp)
            fitted = OUK.variogram_model_parameters
            print("Params:", fitted)

    for nbin in nbins:
        for c in range(nconf):
            rmse = [None for i in range(ntime)]

            for t_chunk in t_chunks:

                pbar.update()

                threads = []
                for t in t_chunk:
                    threads.append(threading.Thread(target=reconstruct, args=(t, args.data_type, grid_lat, grid_lon, mask,
                                                                              model, params[c], nbin,
                                                                              varplot, search)))
                    threads[-1].start()
                for thread in threads:
                    thread.join()

            tot_rmse = sum(rmse)
            print("RMSE:", tot_rmse)
            if search:
                if tot_rmse < rmin:
                    rmin = tot_rmse
                    optim = params[c]
                    onbin = nbin


    if search:
        print("* Best RMSE: ", rmin)
        print("* Best nbin: ", onbin)
        print("* Best params: ", optim)
    else:
        outname = ".".join(args.data_name.split(".")[:-1]) + "_kriged.nc"
        ds.to_netcdf(args.output_dir + outname)

if __name__ == "__main__":
    nckrig()
