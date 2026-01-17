import sys
sys.path.insert(0, '.')
from services.sample_service import get_sample_service

try:
    print("Creating sample service...")
    service = get_sample_service()
    print("Service created successfully!")
    
    print("\nGetting samples...")
    samples = service.get_samples(5, 5)
    print(f"Got {len(samples)} samples")
    
    asd_count = sum(1 for s in samples if s['type'] == 'asd')
    control_count = sum(1 for s in samples if s['type'] == 'control')
    print(f"ASD samples: {asd_count}")
    print(f"Control samples: {control_count}")
    
    if len(samples) > 0:
        print(f"\nFirst sample: {samples[0]}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
