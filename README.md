# nibble4 — 分块 4 位低精度数值库（纯 Rust / 纯 CPU）

`nibble4` 用一个**明确的 16 码表**研究四位表示对点积和矩阵乘法的影响。
仅依赖 Rust 标准库（第三方依赖数为 **0**，上限 2 以内），可在普通 Linux CPU
上用 `cargo build` 构建。

## 码表（16 项）

每个半字节为 `[符号位 | 低三位索引]`。低三位对应幅度：

| 低三位 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-------|---|---|---|---|---|---|---|---|
| 幅度  | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

- `0x0..=0x7` 为非负值，`0x9..=0xF` 为对应负值。
- `0x0` 是正零；`0x8` 是**可读的负零**：解码得到 `-0.0`，但编码器永不写出
  负零（`-0.0` 一律写成 `0x0`）。

## 分块量化

- 块长可配置（`QuantConfig::new(block_len, threshold)`）。
- 每块按最大绝对值映射到幅度 `6`：`scale = max_abs / 6`。
- 舍入为取最近码值；**等距时选绝对值更小者**（向零的半数规则）。
- 全零块具有确定性尺度 `scale = 0.0`，码值全为 0。
- 末尾不足一块的尾块使用自己的尺度，可完整往返。

## 半字节打包规则

- 常量 `LOW_NIBBLE_FIRST = true`：元素 `i` 为偶数时占**低半字节**，奇数时占高半字节。
- 常量 `ODD_TAIL_PAD = 0x0`：奇数个编码时，末字节高半字节填 `0x0`；
  解码若发现该填充非 0 会报 `Corrupted`。
- 公开函数 `pack_nibbles` / `unpack_nibbles`。

## 裁剪与稀疏补偿

- `threshold` 为可配置的绝对值阈值；`f64::INFINITY` 关闭裁剪。
- 先裁剪再量化；残差 `r = x - clipped(x)` 只对被裁剪元素以
  `(u32 索引, f64 残差)` 稀疏存储（索引升序）。
- 残差**实际参与点积和矩阵乘法**，不是只写进压缩文件：
  `use_compensation = true` 时把残差加回操作数，`false` 时忽略残差。
  两种模式对应**完全相同的已编码输入和参数**，只切换该布尔值。

## 存储字节布局（`QVector::to_bytes` / `from_bytes`）

| 区域 | 字节 | 说明 |
|------|------|------|
| magic | 4 | `QNP1` |
| n | 4 LE | 元素个数 |
| block_len | 4 LE | 块长 |
| threshold | 8 LE | f64 |
| flags | 1 | bit0 = 存在残差，其余必须为 0 |
| scales | 8 × 块数 | 每块 f64 尺度 |
| codes | `ceil(n/2)` | 打包半字节 |
| residual_count | 4 LE | 残差对数 |
| residual_idx | 4 × count | 升序 u32 |
| residual_val | 8 × count | f64 残差 |

解析是**全有或全无**：截断、magic/flags 错误、尺度非法（NaN/负数/无穷/与零块矛盾）、
填充非 0、残差索引越界/乱序/与阈值不符等都返回 `Err(QError::...)`，
不会返回部分结果。非有限输入、块长 0、负阈值/NaN、矩阵形状冲突同样被拒绝。

`storage_bytes()` 分项统计实际存储字节（header/scales/codes/residuals/total），
与误差指标是**两回事**：存储量回答“占多少字节”，误差回答“算得有多准”。

## 主要接口

- `QuantConfig::new / no_clipping`
- `QVector::encode / decode / value_at / to_bytes / from_bytes / storage_bytes / len`
- `pack_nibbles / unpack_nibbles / code_to_value / MAGNITUDES`
- `dot(&QVector, &QVector, use_compensation)`；`reference_dot(&[f64], &[f64])`
- `Matrix::new / reference_mul`（纯 f64 参考）；`QMatrix::encode / mul / decode`
- `metrics(reference, approx) -> ErrorMetrics { max_abs_error, mse }`

## 构建、演示与测试

```bash
cargo build
cargo run --example demo      # 小规模演示：码表、打包、补偿开关、点积、矩阵乘
cargo test                    # 实际运行的 11 个集成测试
```

测试断言（非启动/文件存在性检查）：

1. `full_sixteen_entry_code_table` — 16 个码值全覆盖、负零只可读。
2. `round_nearest_tie_picks_smaller_magnitude` — 各中点舍入选较小幅度。
3. `tail_block_and_odd_codes_round_trip` — 尾块、奇数编码与字节往返。
4. `sparse_residual_compensation_contribution` — 残差改变点积数值、存储分项统计。
5. `matmul_matches_independent_reference` / `..._clipped_...` — 矩阵乘与独立 f64 参考一致。
6. 另含半字节打包/填充、非有限与损坏/截断拒绝、误差界与确定性随机性质测试。
