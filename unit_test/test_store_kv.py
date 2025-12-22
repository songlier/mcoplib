import torch
import numpy as np
import unittest
import random
from mcoplib.op import store_kv_cache_cuda_interface

def generate_prefill_session_cache_data(
    batch_size, 
    target_q_len, 
    aver_cache_len
):
    # random q_len, accum to target_q_len
    aver_q_len = target_q_len // batch_size
    q_len_remainder = target_q_len % batch_size
    q_len_offset = aver_q_len // 10
    q_lens = []
    for i in range(batch_size):
        q_lens.append(aver_q_len + (1 if i < q_len_remainder else 0))
    for i in range(batch_size):
        q_lens[i] += random.randint(-q_len_offset, q_len_offset)
    
    # accum q_lens
    accum_q_lens = [0]
    for i in range(batch_size):
        accum_q_lens.append(accum_q_lens[-1] + q_lens[i])

    # random cache_lens
    cache_lens = [aver_cache_len for _ in range(batch_size)]
    cache_offset = aver_cache_len // 10
    for i in range(batch_size):
        cache_lens[i] += random.randint(-cache_offset, cache_offset)

    # sequential cache_slot_ids
    cache_slot_ids = [i for i in range(batch_size)]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens

# batch_size:1 q_seq_len: 6144 cache_len: 18432 kv_head_num 8 q_head_num:8
class TestKVCacheStorage(unittest.TestCase):
    def setUp(self):
        # 设置随机种子保证可重复性
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.src_type = torch.float16
        self.dst_type = torch.int8
        # 通用参数配置
        self.batch_size = 1
        self.q_head_num = 8
        self.kv_head_num = 1
        self.head_dim = 128
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num
        self.q_seq_len = 4096
        self.cache_len = 28672
        
        self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                generate_prefill_session_cache_data(
                    self.batch_size, 
                    self.q_seq_len, 
                    self.cache_len
                )
        self.max_kv_len = max(self.kv_lens)
        self.num_tokens = sum(self.q_lens)
        # 创建输入张量
        self.packed_qkv = torch.randn(
            self.num_tokens,  # 总token数
            self.total_head_num,
            self.head_dim,
            dtype=self.src_type,
            device='cuda'
        )
        
        # 序列长度配置
        self.k_scale = torch.randn(self.kv_head_num, self.head_dim,  dtype=torch.float32, device='cuda')
        self.v_scale = torch.randn(self.kv_head_num, self.head_dim,  dtype=torch.float32, device='cuda')
        
        # 初始化缓存
        self.k_cache = torch.zeros(
            self.batch_size,  # max_batch_size
            self.kv_head_num,
            self.max_kv_len,
            self.head_dim,
            dtype=self.dst_type,
            device='cuda'
        )
        
        self.v_cache = torch.zeros_like(self.k_cache)
    
    def run_reference_impl(self):
        """原始Python实现的参考版本"""
        packed_qkv = self.packed_qkv.clone()
        k_cache_ref = self.k_cache.clone()
        v_cache_ref = self.v_cache.clone()
        
        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            q_offset = self.accum_q_lens[batch_idx]
            cur_cache_len = self.cache_lens[batch_idx]
            cur_slot_id = self.cache_slot_ids[batch_idx]

            token_start = q_offset
            token_end = q_offset + q_len

            k_head_start = self.q_head_num
            k_head_end = self.q_head_num + self.kv_head_num
            v_head_start = self.q_head_num + self.kv_head_num
            v_head_end = self.q_head_num + self.kv_head_num * 2

            cache_start = cur_cache_len
            cache_end = cur_cache_len + q_len

            # [num_tokens, total_head_num, head_dim]
            # --> [q_len, kv_head_num, head_dim]
            # --> [kv_head_num, q_len, head_dim]
            cur_k = packed_qkv[token_start:token_end, k_head_start:k_head_end].transpose(0, 1)
            cur_v = packed_qkv[token_start:token_end, v_head_start:v_head_end].transpose(0, 1)

            # [max_batch_size, total_head_num, max_seq_len, head_dim]
            # --> [kv_head_num, q_len, head_dim]

            k_cache_ref[cur_slot_id, k_head_start - k_head_start:k_head_end - k_head_start, cache_start:cache_end] = torch.round(
                torch.mul(cur_k, self.k_scale.unsqueeze(1))).type(self.dst_type)
            v_cache_ref[cur_slot_id, v_head_start - v_head_start:v_head_end - v_head_start, cache_start:cache_end] = torch.round(
                torch.mul(cur_v, self.v_scale.unsqueeze(1))).type(self.dst_type)
            
        return k_cache_ref, v_cache_ref
    
    def test_single_batch(self):
        """测试单个batch情况"""
        # 复制输入数据
        k_cache_test = self.k_cache.clone()
        v_cache_test = self.v_cache.clone()
    
        # # 运行CUDA实现
        store_kv_cache_cuda_interface(
            self.packed_qkv,
            torch.tensor(self.q_lens, dtype=torch.int32, device='cuda'),
            torch.tensor(self.accum_q_lens, dtype=torch.int32, device='cuda'),
            torch.tensor(self.cache_lens, dtype=torch.int32, device='cuda'),
            torch.tensor(self.cache_slot_ids, dtype=torch.int32, device='cuda'),
            k_cache_test,
            v_cache_test,
            self.k_scale,
            self.v_scale,
            self.batch_size,
            self.q_head_num,
            self.kv_head_num
        )
        
        # 运行参考实现
        k_cache_ref, v_cache_ref = self.run_reference_impl()
        
        # # 比较结果
        torch.testing.assert_close(
            k_cache_test, 
            k_cache_ref,
            atol=0.5,  # 由于四舍五入操作，允许0.5的绝对误差
            rtol=0.01,
            msg="Key cache mismatch"
        )
        
        torch.testing.assert_close(
            v_cache_test, 
            v_cache_ref,
            atol=0.5,
            rtol=0.01,
            msg="Value cache mismatch"
        )
    
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestKVCacheStorage('test_single_batch'))
    
    # 运行测试
    runner = unittest.TextTestRunner()
    runner.run(suite)