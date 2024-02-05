## diesel

- LightGBM model
- Add new trees every X eras
- Ridge regression to ensemble target predictions together
- Partial neutralization over all features

Note: In runs 0-6, we did NOT save intermediary models, only the final model.  As a result, we cannot rebuild validation predictions or refresh metrics.  Since run 3 is one of my top performing models, I opted to rebuild 3 as run id 7 to estimate it's out-of-sample performance in the 1/2/24+ scoring.
