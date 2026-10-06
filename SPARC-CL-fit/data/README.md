# Data

Place the SPARC input files here (or set `SPARC_DATA` to their folder):

- `MassModels_Lelli2016c.mrt`: SPARC rotation curves and mass models (Lelli, McGaugh & Schombert 2016), available from the SPARC website <https://astroweb.case.edu/SPARC/>.
- `Table1wbulge.mrt`: SPARC Table 1 with an added bulge-luminosity column, used for the galaxy properties.

The two Table 1 files are whitespace-delimited. Their byte-by-byte header does not match the data layout, so they are read by splitting on whitespace.
