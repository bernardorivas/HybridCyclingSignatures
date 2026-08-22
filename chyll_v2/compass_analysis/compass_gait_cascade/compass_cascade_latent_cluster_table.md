This table reports a DBSCAN return-map diagnostic only.  It does not measure
cycling signatures or establish topology preservation; clustering the raw
return-map states gives the same period-2, -4, and -8 counts.  The DBSCAN
tolerance is a single fixed fraction of the latent diameter, not a tolerance
sweep.

| label | phi (deg) | expected | DBSCAN clusters | section crossings | noise |
|---|---:|---:|---:|---:|---:|
| phi_1 | 4.75 | 2 | 2 | 2 | 0 |
| phi_2 | 5.00 | 4 | 4 | 4 | 0 |
| phi_3 | 5.02 | 8 | 8 | 8 | 0 |
| phi_4_cloud | 5.20 | chaos | many | many | 65 |
