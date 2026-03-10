from flash_attn import flash_attn_func

def flash_attention(q,k,v):

    return flash_attn_func(q,k,v)