"""Reproduce structural FFN down-projection and Amdahl upper bounds.
Analytical only: no pretrained execution, accuracy, energy or hardware data.
Source G5-R: 8510d89980816e06a511ab0d665188cbe7da3d6c
"""
CONFIGS={"qwen05":(896,4864,24,14,2,151936),
         "granite4micro":(2560,8192,40,40,8,100352)}
TIMING={"qwen05_b4_T32000":(0.5861178531211508,0.5264615314930605),
        "granite4micro_b1_T16384":(0.7052066730549075,0.5023158439279299)}

def count(shape):
    d,m,L,hq,hkv,v=shape
    assert d%hq==0 and hq%hkv==0 and min(shape)>0
    down=L*d*m
    all_weights=L*(2*d*d+2*d*(hkv*d//hq)+3*d*m)+v*d
    return all_weights,down,all_weights/(all_weights-down)

def frontier(f,r):
    assert 0<=f<1 and 0<=r<=1
    return 1/f if f else float("inf"), max(0,(f*r+(1-f)-0.5)/(1-f))

def test():
    assert count(CONFIGS["qwen05"])[:2]==(493961216,104595456)
    assert count(CONFIGS["granite4micro"])[:2]==(3402629120,838860800)
    assert all(count(v)[2]<2 for v in CONFIGS.values())
    for f,r in TIMING.values():
        assert f>0.5
        s,w=frontier(f,r)
        assert s<2 and 0.5<w<0.55
        assert abs(f*r+(1-f)*(1-w)-0.5)<1e-12

if __name__=="__main__":
    test()
    for k,c in CONFIGS.items(): print(k, count(c))
    for k,(f,r) in TIMING.items(): print(k, frontier(f,r))
    print("Structural replay PASS; full 68-test results are separate")
