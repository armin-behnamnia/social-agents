import os
import io
import sys
import numpy as np
import pandas as pd
from Melodie import db
from Melodie import Config

if os.environ.get('MPLBACKEND') == 'module://matplotlib_inline.backend_inline':
    os.environ['MPLBACKEND'] = 'agg'  # یا 'tkagg'
import matplotlib
import matplotlib.pyplot as plt

# تنظیم encoding برای خروجی کنسول
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
from pathlib import Path

class PreferenceAnalyzer:
    def __init__(self, db):
        self.db = db
        self.data_dir = "data/output"
        
    def plot_health_state_share(self, id_scenario=0):

        try:

        # -----------------------------------------
        # Load data
        # -----------------------------------------

            try:
                df = self.db.read_dataframe("environment_result")
                print("Data loaded from database")

            except AttributeError:
                print(
                    "Database method not found, "
                    "trying to read from CSV..."
                )
                df = self._load_from_csv()

            except Exception as e:
                print(f"Error reading from database: {e}")
                df = self._load_from_csv()


        # -----------------------------------------
        # Check data
        # -----------------------------------------

            if df is None or df.empty:
                print("No data available!")
                print("Please run the simulation first:")
                print("python run_simulator.py")
                return


        # -----------------------------------------
        # Filter scenario
        # -----------------------------------------

            if 'id_scenario' in df.columns:
                df = df[df['id_scenario'] == id_scenario]
 

            if df.empty:
                print(
                    f"No data found for scenario {id_scenario}"
                )
                return


        # -----------------------------------------
        # Information
        # -----------------------------------------

            print(
                f"\nData shape for scenario {id_scenario}: "
                f"{df.shape}"
            )

            print(
                "Columns:",
                df.columns.tolist()
            )

            print("\nFirst few rows:")
            print(df.head())


        # -----------------------------------------
        # Plot
        # -----------------------------------------

            self._plot_data(
                df,
                id_scenario
            )


        except Exception as e:
  
            print(
                f"Error processing scenario "
                f"{id_scenario}: {e}"
            )

            import traceback
            traceback.print_exc()
    
    def _load_from_csv(self):
        """بارگذاری داده از فایل CSV"""
        csv_paths = [
            os.path.join(self.data_dir, "Result_Simulator_Environment.csv"),
            os.path.join("..", self.data_dir, "Result_Simulator_Environment.csv"),
            os.path.join("Desktop/LAIDSim-master", self.data_dir, "Result_Simulator_Environment.csv"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), self.data_dir, "Result_Simulator_Environment.csv")
        ]
        
        for path in csv_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    print(f"Data loaded from: {path}")
                    return df
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue
        
        print("CSV file not found in any of the expected locations!")
        return None
    
    def _plot_data(self, df, id_scenario):
        """رسم نمودار از داده‌ها"""
        try:
            # ستون‌های موجود را شناسایی کنید
            print(f"\nPlotting data for scenario {id_scenario}...")
            
            # بررسی وجود ستون‌های مورد نیاز
            health_columns = ['s0', 's1']  # ستون‌های وضعیت سلامتی
            time_column = 'period'  # ستون زمان/مرحله
            
            # اطمینان از وجود ستون‌ها
            if time_column not in df.columns:
                print(f"Column '{time_column}' not found!")
                print("Available columns:", df.columns.tolist())
                return
            
            # بررسی ستون‌های سلامت
            available_health_cols = [col for col in health_columns if col in df.columns]
            if not available_health_cols:
                #print("No health state columns found!")
                #print("Available columns:", df.columns.tolist())
                return
            
            # گروه‌بندی بر اساس period و محاسبه میانگین برای هر وضعیت سلامت
            grouped_df = df.groupby(time_column)[available_health_cols].mean()
            
            # ایجاد نمودار
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # رسم هر ستون به عنوان یک خط جداگانه
            for col in available_health_cols:
                ax.plot(grouped_df.index, grouped_df[col], marker='o', label=f'{col}')
            
            ax.set_xlabel('Period')
            ax.set_ylabel('Value')
            ax.set_title(f'Health State Distribution - Scenario {id_scenario}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # ذخیره نمودار
            plot_path = PLOTS_DIR / f"health_state_scenario_{id_scenario}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            #print(f"Plot saved to: {plot_path}")
            #print(f"File exists: {plot_path.exists()}")
            #print(f"File size: {plot_path.stat().st_size} bytes")
            plt.close()

        except Exception as e:
            print(f"Error processing data: {e}")
            import traceback
            traceback.print_exc()
    def _plot_stacked_area(self, df, id_scenario, health_cols, time_column):
        """رسم نمودار stacked area برای نمایش سهم هر وضعیت"""
        try:
            # گروه‌بندی بر اساس period
            grouped_df = df.groupby(time_column)[health_cols].mean()
            
            # محاسبه درصدی
            total = grouped_df.sum(axis=1)
            for col in health_cols:
                grouped_df[f'{col}_pct'] = (grouped_df[col] / total) * 100
            
            # ستون‌های درصدی
            pct_cols = [f'{col}_pct' for col in health_cols]
            
            # رسم stacked area
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.stackplot(grouped_df.index, 
                        [grouped_df[col] for col in pct_cols],
                        labels=health_cols,
                        alpha=0.7)
            
            ax.set_xlabel('Period')
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Health State Share - Scenario{id_scenario}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # ذخیره نمودار
            plot_path = PLOTS_DIR / f'health_state_share_scenario_{id_scenario}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            #print(f"Stacked area plot saved to: {plot_path}")
            #print(f"File exists: {plot_path.exists()}")
            #print(f"File size: {plot_path.stat().st_size} bytes")

            plt.close()
            
            # نمایش نمودار
            # plt.show()
            
        except Exception as e:
            print(f"Error creating stacked area plot: {e}")

    def plot_alteration_degree(self, id_scenario=0):
        """رسم نمودار درجه تغییرات"""
        try:
            df = self._load_from_csv()
            if df is None or df.empty:
                #print("No data available!")
                return
            
            # فیلتر بر اساس سناریو
            if 'id_scenario' in df.columns:
                df = df[df['id_scenario'] == id_scenario]
            
            if df.empty:
                #print(f"No data found for scenario {id_scenario}")
                return
            
            if 'alteration_degree' not in df.columns:
                #print("Column 'alteration_degree' not found!")
                return
            
            # رسم نمودار
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # نمایش تغییرات برای هر run
            if 'id_run' in df.columns:
                for run_id in df['id_run'].unique():
                    run_data = df[df['id_run'] == run_id]
                    ax.plot(run_data['period'], run_data['alteration_degree'], 
                           marker='o', label=f'Run {run_id}')
            else:
                ax.plot(df['period'], df['alteration_degree'], marker='o')
            
            ax.set_xlabel('Period')
            ax.set_ylabel('Alteration Degree')
            ax.set_title(f'Alteration Degree - Scenario {id_scenario}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # ذخیره نمودار
            plot_path = PLOTS_DIR / f'alteration_degree_scenario_{id_scenario}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            #print(f"Plot saved to: {plot_path}")
            #print(f"File exists: {plot_path.exists()}")
            #print(f"File size: {plot_path.stat().st_size} bytes")
            plt.close()
            
            # plt.show()
            
        except Exception as e:
            print(f"Error plotting alteration degree: {e}")