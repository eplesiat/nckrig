
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
    global ds, rmse, fitted

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
    parser.add_argument("-w", "--n-points", type=int, default=None, help="Number of nearby points to use with a moving window")
    parser.add_argument("-k", "--backend", type=str, default="vectorized", help="Backend for the execute")
    parser.add_argument("-n", "--n-threads", type=int, default=1, help="Number of threads")
    parser.add_argument('--steady-mask', type=str, default=None,help="NetCDF file containing a single mask"
                                                                     "to be applied to all timesteps.")
    parser.add_argument('--data-dir', type=str, default='',
                                help="Directory containing the climate datasets")
    parser.add_argument('--mask-dir', type=str, default='', help="Directory containing the mask datasets")
    parser.add_argument('--output-dir', type=str, default='',
                                help="Directory where the output files will be stored")
    parser.add_argument('--output-name', type=str, default=None, help="Prefix used for the output filename")
    parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile, help="Load all the arguments from a text file")
    args = parser.parse_args()

    model = args.model

    if args.universal:
        kriging = UniversalKriging
    else:
        kriging = OrdinaryKriging

    ds = xr.open_dataset(args.data_dir + args.data_name)
    ds2 = ds.copy()

    if args.mask_name is None:
        mask = None
    else:
        mask = (1 - xr.open_dataset(args.mask_dir + args.mask_name)[args.data_type].values).astype(bool)
        assert ds[args.data_type].shape == mask.shape

    if args.steady_mask is None:
        steady_mask = None
    else:
        steady_mask = (1 - xr.open_dataset(args.mask_dir + args.steady_mask)[args.data_type].values).astype(bool).squeeze()
        assert ds[args.data_type].shape[1:] == steady_mask.shape

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
    t_chunks = np.array_split(np.random.permutation(ntime), args.n_threads)
    print("* Chunks: ", t_chunks)

    nconf = len(params)
    ntot = nconf * len(nbins)

    print("* Total number of configurations:", ntot)

    search = False
    varplot = args.varplot
    if ntot > 1:
        search = True
        varplot = False

    pbars = [tqdm(total=ntot * len(t_chunk)) for t_chunk in t_chunks]

    dims = {}
    for dim in ds.dims:
        if "lon" in dim:
            dims["lon"] = dim
        if "lat" in dim:
            dims["lat"] = dim

    grid_lon = ds[dims["lon"]].values
    grid_lat = ds[dims["lat"]].values

    rmin = np.inf

    def reconstruct(pbar, t_chunk, selvar, grid_lat, grid_lon, mask, steady_mask, model, params, nbin, backend, n_points, varplot, search):
        global ds, rmse, fitted

        for t in t_chunk:
            pbar.update()

            vals = ds[selvar].values[t].copy()
            if mask is not None:
                vals[mask[t]] = np.nan

            idx = np.where(~np.isnan(vals))
            vals = vals[idx]
            lat = grid_lat[idx[0]]
            lon = grid_lon[idx[1]]

            try:
                OUK = kriging(np.array(lon), np.array(lat), np.array(vals), variogram_model=model, verbose=False,
                                    variogram_parameters=params, nlags=nbin, enable_plotting=varplot)
                interp, ss1 = OUK.execute('grid', grid_lon, grid_lat, backend=backend, n_closest_points=n_points)

            except:
                print("Warning! Could not reconstruct data for t = ", t)
                interp = np.ones_like(ds[args.data_type][t].values) * np.nanmean(vals)
                ss1 = np.zeros_like(interp)

            if steady_mask is not None:
                interp[steady_mask] = np.nan
                ss1[steady_mask] = np.nan

            rmse[t] = np.sqrt(np.nanmean((ds[args.data_type][t].values - interp)**2))

            if not search:
                ds[args.data_type][t] = np.array(interp)
                ds2[args.data_type][t] = np.array(ss1)
                fitted[t] = OUK.variogram_model_parameters

    for nbin in nbins:
        for c in range(nconf):
            rmse = [np.nan for i in range(ntime)]
            fitted = [np.nan for i in range(ntime)]

            threads = []
            k = 0
            for t_chunk in t_chunks:
                threads.append(threading.Thread(target=reconstruct, args=(pbars[k], t_chunk, args.data_type, grid_lat, grid_lon,
                                                                          mask, steady_mask, model, params[c], nbin,
                                                                          args.backend, args.n_points, varplot, search)))
                threads[-1].start()
                k += 1

            for thread in threads:
                thread.join()

            tot_rmse = np.nanmean(rmse)
            print("RMSE:", tot_rmse)
            if search:
                if tot_rmse < rmin:
                    rmin = tot_rmse
                    optim = params[c]
                    onbin = nbin
            # else:
            #     print("Params:")
            #     for t in range(ntime):
            #         print(t, fitted[t])

    if search:
        print("* Best RMSE: ", rmin)
        print("* Best nbin: ", onbin)
        print("* Best params: ", optim)
    else:
        if args.output_name is None:
            outname = ".".join(args.data_name.split(".")[:-1])
        else:
            outname = args.output_name
        ds.to_netcdf(args.output_dir + outname + "_infilled.nc")
        ds2.to_netcdf(args.output_dir + outname + "_variance.nc")

if __name__ == "__main__":
    nckrig()
