# Getting the Dataset for Training

To train the model, you need the Chest X-Ray Pneumonia dataset.

## Quick Setup (Choose One):

### Option 1: Manual Download from Kaggle (Easiest)
1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download" (you'll need a Kaggle account - it's free)
3. Extract the ZIP file
4. Run this command to organize it:
   ```bash
   python3 setup_data.py /path/to/extracted/dataset
   ```

### Option 2: Using Kaggle API
1. Get your Kaggle API credentials from: https://www.kaggle.com/settings
2. Save the `kaggle.json` file to `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip
   python3 setup_data.py chest-xray-pneumonia
   ```

### Option 3: If You Already Have the Dataset
If you've already downloaded the dataset somewhere, just run:
```bash
python3 setup_data.py /path/to/your/extracted/dataset
```

The script will automatically organize it into the correct structure.

---

Once the dataset is organized in the `data/` directory, you can train with:
```bash
python3 train.py --data_dir data --epochs 10
```

