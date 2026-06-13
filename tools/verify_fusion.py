
import sys
from app.services.models.hybrid.fusion import LateFusionWrapper

def test_late_fusion():
    print("Testing LateFusionWrapper...")
    wrapper = LateFusionWrapper("test_fusion", {"alpha": 0.6})
    
    # Test 1: Tuple Input
    p1 = (0.9, 0.1) # Img=0.9, Snr=0.1. Alpha=0.6. Res = 0.6*0.9 + 0.4*0.1 = 0.54 + 0.04 = 0.58
    res1 = wrapper.predict(p1)
    print(f"Test 1 (Tuple): Input={p1}, Alpha=0.6, Expected=0.58, Got={res1}")
    assert abs(res1 - 0.58) < 1e-6
    
    # Test 2: Dict Input
    p2 = {'image_prob': 0.2, 'sensor_prob': 0.8}
    res2 = wrapper.predict(p2)
    # 0.6*0.2 + 0.4*0.8 = 0.12 + 0.32 = 0.44
    print(f"Test 2 (Dict): Input={p2}, Alpha=0.6, Expected=0.44, Got={res2}")
    assert abs(res2 - 0.44) < 1e-6

    print("SUCCESS: LateFusionWrapper verified.")

if __name__ == "__main__":
    test_late_fusion()
