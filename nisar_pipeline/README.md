# NASA-ISRO NISAR Geospatial Preprocessing Pipeline

Stream-first Python pipeline that pulls AOI windows from ~7 GB NISAR L2/L3 HDF5 granules on ASF DAAC without downloading whole files, computes calibrated radar products in native UTM, and exports lightweight Cloud-Optimized GeoTIFFs, Zarr stacks, and JSON catalogs for web visualization.
