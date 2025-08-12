竞赛来源：https://www.kaggle.com/competitions/china-real-estate-demand-prediction

## 1) 任务与目标

- **任务**：预测各 **sector × month** 的「**新房成交总金额**」（单位：**万元**）。
- **预测目标列**：`train/new_house_transactions.csv` 中的 `amount_new_house_transactions`。
- **注意空缺代表 0**：`train/new_house_transactions.csv` 并不包含所有 `month × sector` 组合；例如不存在 `2019 Jan_sector 3` 的记录，**意味着真实值为 0**，在测试集也会出现这种需要预测为 0 的样本。

------

## 2) 评估方式（两阶段自定义分数）

记
$$
\mathrm{APE}_i=\frac{|y_i^{\text{pred}}-y_i^{\text{true}}|}{y_i^{\text{true}}}
$$
为第 i 个样本的相对误差（绝对百分比误差）。

### 阶段一（硬门槛）

若
$$
\frac{1}{n}\sum_{i=1}^n \mathbf{1}\{\mathrm{APE}_i>1\} \;>\; 0.3
$$
则**分数直接为 0**。——也就是**超过 30% 的样本相对误差大于 100%** 时，成绩清零。

### 阶段二（通过门槛后计分）

令
$$
D=\Big\{\mathrm{APE}_i \;\big|\; \mathrm{APE}_i\le 1\Big\},\quad |D|\text{ 为样本数}
$$
计算
$$
\text{scaled\_MAPE} =\frac{\operatorname{average}(D)}{|D|/n} \quad\Longrightarrow\quad \text{score}=1-\text{scaled\_MAPE}.
$$
**实现细节建议（本地验证）**

- 为避免
  $$
  y^{\text{true}}=0
  $$
  的除零不稳定，线下可用 
  $$
  \mathrm{APE}_i=\frac{|y^{\text{pred}}-y^{\text{true}}|}{\max(y^{\text{true}},\varepsilon)}
  $$
  （如 
  $$
  \varepsilon=1\mathrm{e}{-9}
  $$
  ）。

- 当 
  $$
  y^{\text{true}}=0
  $$
   时：若预测为 0，应视为
  $$
  \mathrm{APE}=0
  $$
  （进入集合 $D$）；否则 $\mathrm{APE}$ 视为很大（>1），会贡献到阶段一的「坏样本比例」。

### 对建模/损失的含义

- **首要目标**：尽量减少 
  $$
  \mathrm{APE}>1
  $$
   的比例（>30% 会直接 0 分）。

- **次要目标**：在
  $$
  \mathrm{APE}\le 1
  $$
   的子集内让相对误差尽可能小，同时保证该子集占比 
  $$
  |D|/n
  $$
   较大（否则 scaled_MAPE 会被放大）。

**可选的训练度量/损失替代：**

- **对数域回归**：对目标做 `log1p` 回归，预测后 `expm1`，天然抑制极端值，降低 
  $$
  \mathrm{APE}>1
  $$
   的风险。

- **相对误差型损失（线下）**：最小化
  $$
  MAPEε=1n∑i∣yi−y^i∣max⁡(yi,ε)\mathrm{MAPE}_\varepsilon=\frac{1}{n}\sum_i\frac{|y_i-\hat y_i|}{\max(y_i,\varepsilon)}
  $$
  或 **SMAPE**；并对
  $$
  \mathrm{APE}>1
  $$
  的样本**加大权重**（例如权重 3~5 倍），相当于软地优化阶段一门槛。

- **稳健裁剪/回退**：以 **lag-1** 或滑窗均值为回退参考，对预测值做上下限裁剪（如 
  $$
  [{\text{fallback}}/3,\, 3\times{\text{fallback}}]
  $$
  ），可显著压降大比例「翻车样本」。

------

## 3) 数据与文件结构

- `train/*.csv`：训练期可用的全部特征表（见下）。
- `test.csv`：测试集索引（行顺序即提交顺序）。
- `sample_submission.csv`：提交示例。
- **泄露约束**：**预测某月时，禁止使用任何未来月份的数据**（包括「近邻 sector」表中的未来观测）。

### 3.1 目标与核心交易表

**`train/new_house_transactions.csv`**（核心，含目标）

- `month`：月份（字符串，如 `YYYY Mon`；实际可与 sector 合并形成 `id`）。
- `sector`：地理分区（整数或字符串形如 `sector n`）。
- `num_new_house_transactions`：新房成交套数。
- `area_new_house_transactions`：新房成交面积（平方米）。
- `price_new_house_transactions`：新房成交均价（元/㎡）。
- `amount_new_house_transactions`：**新房成交总金额（万元）—目标列**。
- 其余派生指标：如 `area_per_unit_*`、`total_price_per_unit_*`、`*_available_for_sale`、`period_new_house_sell_through` 等。

**邻近分区版：`train/new_house_transactions_nearby_sectors.csv`**

- 将上述多列替换为「…`_nearby_sectors`」，刻画周边影响。

### 3.2 二手房交易表

- **`train/pre_owned_house_transactions.csv`**：本 sector 的二手房成交面积/金额/套数/均价。
- **`train/pre_owned_house_transactions_nearby_sectors.csv`**：周边 sector 的对应聚合。

### 3.3 土地交易表

- **`train/land_transactions.csv`**：本 sector 的土地成交（宗数、建筑面积、成交额等）。
- **`train/land_transactions_nearby_sectors.csv`**：周边 sector 的对应聚合。

### 3.4 静态/准静态区位特征

- **`train/sector_POI.csv`**：POI & 人口与商业密度画像（大量计数与密度字段，含交通/教育/医疗/商业等细分维度），以及周边房价/租金等。

### 3.5 城市级搜索与宏观指标

- **`train/city_search_index.csv`**：`month, keyword, source, search_volume`（可能需按关键词做分组/筛选/宽表透视）。
- **`train/city_indexes.csv`**：年度级城市宏观与财政、产业、教育医疗、基建等指标（需按年对齐到月：如同年复制或用年内所有月同值）。

------

## 4) ID 与提交规范

- **`id` 组成**：`"%Y %b_sector n"`，如 `2024 Aug_sector 3`。

- **提交列**：**恰好两列**

  - `id`
  - `new_house_transaction_amount`（预测值，单位万元；非负）

- **行顺序**：**必须保持与 `test.csv` 完全一致**。

- 官方样例：

  ```
  id,new_house_transaction_amount
  2024 Aug_sector 1,49921.00868
  2024 Aug_sector 2,92356.22985
  2024 Aug_sector 3,480269.7819
  ...
  ```

------

## 5) 数据泄露与特征工程要点

**硬规则**：**任何滞后/滑窗/同比环比统计都只能使用“过去与当月”信息**；预测月份 t不得看 t+1及以后。

**推荐做法**

- **时间索引**：把 `month` 解析为 `date`（月首），构造 `year, month, quarter, weekofyear, month_idx`。
- **分组维度**：至少按 `sector` 分组做滞后与滚动（`lag_1, lag_2, lag_3, lag_6`；`roll_mean_3/6` 等）。
- **跨表对齐**：
  - 以 `['date','sector']` 为键左连接：
    - 新房本表 & 周边表
    - 二手房本表 & 周边表
    - 土地本表 & 周边表
    - `sector_POI`（静态 → 直接按 `sector` 连接）
    - `city_search_index`（需先对 `keyword` 做挑选/聚合，如与“新房/楼盘/买房/房价”相关的关键词窗口和；按 `date` 汇总）
    - `city_indexes`（年度 → 映射到每个月，同年赋值）
- **派生特征**（示例）
  - 新房：`amount = area × price` 已在表内；可做 `price_yoy/mom`、`area_yoy/mom`、`sell_through_period` 的区间缩放。
  - **二手 ↔ 新房**：价差、量差、价量比（本 sector 与周边对比）。
  - **土地 → 未来供给**：`planned_building_area` 与 `transaction_amount` 的滞后对后续 6~12 个月新房成交的影响。
  - **POI 密度**：做对数/标准化；与 `sector_coverage`、人口规模交互。
- **零值处理**：训练集中缺失的 `month×sector` 组合等价于 **目标=0**；做滞后/滑窗时，缺历史则产生 `NaN`，由中位数/均值填充或以 sector 级中位数回退。

------

## 6) 交叉验证与线下度量

- **切分方式**：时间序列切分（`TimeSeriesSplit` 或逐月 Walk-Forward），**严格按时间前训练、后验证**。
- **线下评估**：实现与官方一致的**两阶段评分**函数，用于早停与模型比较；同时保留 RMSE/MAE 作为参考。
- **目标变换**：建议提供开关 `log1p(y)`；对长尾金额更稳定。
- **后处理**：
  1. 非负裁剪：`pred = max(pred, 0)`
  2. 与 `lag_1 / roll_mean_3` **融合与裁剪**，降低 $\mathrm{APE}>1$ 的比例，守住阶段一门槛。
  3. 对预期为 0 的组合，优先输出 0（若基线与统计均支持），避免无谓失分。

------

## 7) 快速 Baseline（思路提要）

1. 读入 `train/*` 表；按 `['date','sector']` 纵向合并为**月度宽表**。
2. 仅用历史构建 `lag/rolling`；连接静态与城市级特征。
3. `TimeSeriesSplit` 做 CV；模型用 **LightGBM 回归**（或 XGBoost/CatBoost）；目标可 `log1p`。
4. 训练时监控自定义**两阶段评分**（线下实现），并打印「坏样本占比」与分数。
5. 预测后做非负与稳健裁剪；生成提交 `id,new_house_transaction_amount`，**保持 test.csv 行序**。

------

## 8) 提交前检查清单

-  只使用到 **预测月份及之前** 的数据（包含「周边 sector」表）。
-  输出两列：`id,new_house_transaction_amount`；单位**万元**；值**非负**。
-  行顺序与 `test.csv` 完全一致；`id` 字符串格式 `"%Y %b_sector n"`。
-  线下两阶段评分 **阶段一坏样本占比 ≤ 0.30**；否则即使线上其他指标好也会**0 分**。
-  代码可复现（组织者会向前 10 名索取代码，并可能用 2025 未公布数据复测）。