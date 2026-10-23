# Component Inspection UI

A PySide6 desktop UI for browsing component types, part numbers, inspection batches, individual inspected items, PDFs, part specifications, and cross-part metric comparisons.

## Run it

1. Install Python 3.10 or newer.
2. In this directory, install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   python main.py
   ```

4. Choose **File → Load inspection data (bolts, nuts, IC packages)** to create sample batches and individual component PDFs.

The default data root is this directory. To use a different location, set `INSPECTION_DATA` or use **File → Choose data folder…**. Generated data stays organized by component type, part number, batch, and individual component; it does not create visible `inspection_data` or `bolt_results` folders by default.

## What changed

- The batch table includes **Total items** and **Items rejected**. Batch and component results use only **Conform** and **Non-conform**. The component document button opens the PDF for the selected item in the batch.
- The **Inspected items** tab shows expected and read values for each individual item. Out-of-tolerance measurements are highlighted. Select a row and click **View selected component document** to open its individual PDF.
- Each inspected item has its own PDF under `component type / part number / batch / components/`.
- Charts compare the latest available batch metrics across part numbers, rather than plotting batches over time.
- Sample inspection data includes four bolt part numbers, three nut part numbers, and three IC package/chip part numbers. Each has two batches of eight synthetic inspected items with some deliberately out-of-tolerance examples.

## Notes about data

- Sample records are **generated test data**, not actual measurements or certified engineering tolerances. Edit each part's specification and tolerance in **Part spec** before using the app for real quality decisions.
- The model-backed scanner test in `pdf.py` currently models the bundled bolt datasheet part `91247A724`. The expanded bolts/nuts/chips catalog is generic sample data; it does not imply that the bolt-verification model has been trained or validated for those other components.
- For real batches, the pipeline's `parts.csv` is used to populate item-level values. Older batch folders that contain only aggregate summaries cannot recover historical per-item measurements; the UI marks those generated per-item values as estimates in the record data.
