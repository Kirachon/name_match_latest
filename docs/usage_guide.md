# Usage Guide — How to Use the Matching

## CLI Usage

### Basic Syntax

```
name_matcher <host> <port> <user> <password> <database> <table1> <table2> <algo> <out_path> [format] [flags...]
```

### Parameters

| Position | Name | Description |
|----------|------|-------------|
| 1 | host | MySQL host address |
| 2 | port | MySQL port (e.g., 3306) |
| 3 | user | Database username |
| 4 | password | Database password |
| 5 | database | Database name |
| 6 | table1 | Source table name |
| 7 | table2 | Target table name |
| 8 | algo | Algorithm number (1–7) |
| 9 | out_path | Output file path |
| 10 | format | `csv`, `xlsx`, or `both` (default: csv) |

### Quick Start Examples

```bash
# Deterministic match (Algo 1), CSV output
name_matcher 127.0.0.1 3306 root secret mydb persons targets 1 D:/out/matches.csv

# Fuzzy match (Algo 3), XLSX output
name_matcher 127.0.0.1 3306 root secret mydb persons targets 3 D:/out/matches.xlsx xlsx

# Both formats with GPU hash join
name_matcher 127.0.0.1 3306 root secret mydb persons targets 1 D:/out/matches both --gpu-hash-join
```

### Using Environment Variables Instead

```powershell
$env:DB_HOST="127.0.0.1"
$env:DB_PORT="3306"
$env:DB_USER="root"
$env:DB_PASSWORD="secret"
$env:DB_NAME="mydb"
$env:TABLE1="persons"
$env:TABLE2="targets"
$env:ALGO="3"
$env:OUT_PATH="D:/out/matches.csv"

name_matcher
```

### Generate .env Template

```bash
name_matcher env-template [path]
```

---

## GUI Usage

### Launching

```bash
# Windows
target\release\gui.exe

# Linux
./target/release/gui
```

### Step-by-Step Workflow

#### 1. Configure Database Connection

- Fill in Host, Port, Username, Password, Database fields
- (Optional) Enable "Dual Database" and fill second DB fields
- Click **"Load Tables"** to connect and discover tables

#### 2. Select Tables

- Choose **Table 1** (source) from the dropdown
- Choose **Table 2** (target) from the dropdown

#### 3. Choose Algorithm

Select from the algorithm dropdown:
- **Option 1:** Deterministic (First + Last + Birthdate)
- **Option 2:** Deterministic (First + Middle + Last + Birthdate)
- **Option 3:** Fuzzy (with Middle Name)
- **Option 4:** Fuzzy (without Middle Name)
- **Option 5:** Household Matching (Table1→Table2)
- **Option 6:** Household Matching (Table2→Table1)
- **Option 7:** Levenshtein Weighted

#### 4. Configure Execution Mode

- **Auto:** Application decides based on dataset size and available memory
- **Streaming:** Processes in batches (lower memory, recommended for large datasets)
- **In-Memory:** Loads all records into RAM (faster for small datasets)

#### 5. (Optional) Enable GPU Acceleration

- Check "Use GPU" to enable CUDA acceleration
- Configure GPU-specific options (hash join, fuzzy metrics, streams, etc.)
- Set VRAM budget if needed

#### 6. Set Output

- Choose output file path (use the file picker button)
- Select format: CSV / XLSX / Both
- For fuzzy algorithms, adjust the confidence threshold slider (60–100%)

#### 7. Run Matching

- Click **Start**
- Monitor progress bar, ETA, records/second, memory usage
- Use **Pause** to temporarily halt, **Cancel** to abort

#### 8. Review Results

- Output files are written to the specified path
- Check the log console for summary statistics
- XLSX files include a summary worksheet with timing and match counts

---

## Advanced: Cascade Matching

Enable cascade matching in the GUI to run levels L1–L11 sequentially:

1. Check **"Advanced Matching"** or **"Cascade"** toggle
2. Configure missing column mode:
   - **AutoSkip** (default): Skips levels needing missing columns
   - **ManualSelect**: Choose specific levels
   - **AbortOnMissing**: Fails if geographic columns absent
3. Records matched at earlier levels are excluded from later levels
4. Results include the level at which each match was found

---

## Choosing the Right Algorithm

| Scenario | Recommended Algorithm |
|----------|----------------------|
| High-confidence dedup, middle names unreliable | Algorithm 1 |
| Strict matching with all name parts | Algorithm 2 |
| Spelling variations, transliteration differences | Algorithm 3 |
| Middle names sparse/inconsistent | Algorithm 4 |
| Link individuals to households (T1→T2) | Algorithm 5 |
| Link households to individuals (T2→T1) | Algorithm 6 |
| Fine-grained weighted edit distance | Algorithm 7 |
| Progressive multi-criteria matching | Cascade (L1–L11) |

---

## Understanding Output

### CSV/XLSX Columns

| Column | Description |
|--------|-------------|
| Table1_ID, Table1_UUID | Source record identifiers |
| Table1_FirstName, Table1_LastName, etc. | Source record fields |
| Table2_ID, Table2_UUID | Target record identifiers |
| Table2_FirstName, Table2_LastName, etc. | Target record fields |
| Table2_* (dynamic) | Extra fields from Table 2 |
| is_matched_* | Boolean match flag |
| Confidence | Match confidence score (0–100) |
| MatchedFields | List of fields that matched |

### Fuzzy Confidence Interpretation

| Score | Meaning |
|-------|---------|
| 100% | Exact match (all fields identical after normalization) |
| 90–99% | Very high confidence (minor variations) |
| 85–89% | High confidence (Case 1/2/3 thresholds met) |
| 60–84% | Moderate confidence (review recommended) |
