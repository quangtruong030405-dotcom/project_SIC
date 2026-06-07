# MAVO-LSTM vs PSO-LSTM

Project thuc nghiem toi uu sieu tham so LSTM cho du bao `Global_active_power`
tren bo `household_power_consumption.csv`.

## Chay nhanh

```powershell
C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe main_experiment.py --quick
```

Ket qua duoc luu trong `results/`, gom convergence CSV, bang tham so tot nhat,
bang validation/test va cac hinh SVG.

## Chay cau hinh tuy chinh

```powershell
C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe main_experiment.py --seeds 1 2 3 --population 10 --budget 100 --epochs 10
```

Mac dinh script dung NumPy LSTM mot lop de khong phu thuoc TensorFlow/PyTorch.
Neu can bao cao chinh, tang `--budget`, `--population`, `--epochs` va bo
`--max-hours` de dung toan bo chuoi gio.
