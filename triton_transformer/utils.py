
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_num_warps(block_size):
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    return num_warps
