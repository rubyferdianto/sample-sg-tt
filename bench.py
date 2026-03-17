import time, sys, types

# Mock streamlit with a permissive mock
class _Mock:
    def __getattr__(self, name):
        return _Mock()
    def __call__(self, *a, **kw):
        if any(callable(x) for x in a):
            return a[0]  # decorator passthrough
        return _Mock()

sys.modules['streamlit'] = _Mock()

from streamlit_app import run_ml_model_analysis, load_data

df = load_data()
print(f"Data loaded, {len(df)} rows")

t0 = time.time()
result = run_ml_model_analysis(df)
elapsed = time.time() - t0

print(f"ML analysis completed in {elapsed:.1f}s")
print(f"Training samples: {result['total_training_samples']}, Features: {result['num_features']}")
for name, res in result['results'].items():
    print(f"  {name}: total_matches={res['total_matches']}, mae={res['mae']:.1f}")

print("\n--- Future Predictions (checking for variety) ---")
for name, preds in result['future_preds'].items():
    print(f"\n  {name}:")
    for i, p in enumerate(preds):
        print(f"    Draw {result['test_end']+1+i}: {p}")
