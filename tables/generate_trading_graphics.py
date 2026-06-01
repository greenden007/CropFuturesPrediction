"""
Generate graphics and organized tables from trading_comparison_summary.csv

Usage:
    python generate_trading_graphics.py
    python generate_trading_graphics.py --csv ../trading/trading_results/trading_comparison_summary.csv
    python generate_trading_graphics.py --output_dir ./output
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table."""
    if HAS_TABULATE:
        return tabulate(df, headers='keys', tablefmt='pipe', showindex=False, floatfmt='.2f')
    # Fallback manual implementation
    lines = []
    headers = '| ' + ' | '.join(df.columns) + ' |'
    lines.append(headers)
    lines.append('|' + '|'.join(['---'] * len(df.columns)) + '|')
    for _, row in df.iterrows():
        row_str = '| ' + ' | '.join(str(v) for v in row.values) + ' |'
        lines.append(row_str)
    return '\n'.join(lines)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean trading comparison data."""
    df = pd.read_csv(csv_path)
    # Convert percentage strings to floats
    df['Win Rate'] = df['Win Rate'].str.replace('%', '').astype(float)
    return df


def generate_summary_by_commodity(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics grouped by commodity."""
    summary = df.groupby('Commodity').agg({
        'Return %': ['mean', 'std', 'min', 'max'],
        'Sharpe': ['mean', 'std', 'min', 'max'],
        'Max DD %': ['mean', 'std'],
        'Trades': ['mean'],
        'Win Rate': ['mean'],
        'Profit Factor': ['mean']
    }).round(3)
    
    output_file = output_dir / 'trading_summary_by_commodity.csv'
    summary.to_csv(output_file)
    print(f"Saved: {output_file}")
    return summary


def generate_summary_by_model(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics grouped by model."""
    summary = df.groupby('Model').agg({
        'Return %': ['mean', 'std', 'count'],
        'Sharpe': ['mean', 'std', 'min', 'max'],
        'Max DD %': ['mean', 'std'],
        'Win Rate': ['mean'],
        'Profit Factor': ['mean']
    }).round(3)
    
    output_file = output_dir / 'trading_summary_by_model.csv'
    summary.to_csv(output_file)
    print(f"Saved: {output_file}")
    return summary


def generate_best_performers(df: pd.DataFrame, output_dir: Path):
    """Generate tables of best performers by category."""
    results = {}
    
    # Best by Return % for each commodity
    for commodity in df['Commodity'].unique():
        best_return = df[df['Commodity'] == commodity].nlargest(5, 'Return %')[
            ['Model', 'H', 'Return %', 'Sharpe', 'Win Rate']
        ]
        results[f'best_return_{commodity}'] = best_return
        
        best_sharpe = df[df['Commodity'] == commodity].nlargest(5, 'Sharpe')[
            ['Model', 'H', 'Return %', 'Sharpe', 'Win Rate']
        ]
        results[f'best_sharpe_{commodity}'] = best_sharpe
    
    # Overall best
    results['overall_best_return'] = df.nlargest(10, 'Return %')[
        ['Model', 'Commodity', 'H', 'Return %', 'Sharpe', 'Win Rate', 'Profit Factor']
    ]
    
    results['overall_best_sharpe'] = df.nlargest(10, 'Sharpe')[
        ['Model', 'Commodity', 'H', 'Return %', 'Sharpe', 'Win Rate', 'Profit Factor']
    ]
    
    # Save as markdown
    md_content = "# Trading Performance Best Performers\n\n"
    
    for name, table in results.items():
        title = name.replace('_', ' ').title()
        md_content += f"## {title}\n\n"
        md_content += df_to_markdown(table)
        md_content += "\n\n"
    
    output_file = output_dir / 'best_performers.md'
    with open(output_file, 'w') as f:
        f.write(md_content)
    print(f"Saved: {output_file}")
    
    return results


def plot_returns_by_commodity(df: pd.DataFrame, output_dir: Path):
    """Plot return distributions by commodity."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    commodities = df['Commodity'].unique()
    x_pos = np.arange(len(commodities))
    
    means = [df[df['Commodity'] == c]['Return %'].mean() for c in commodities]
    stds = [df[df['Commodity'] == c]['Return %'].std() for c in commodities]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                   color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
    
    ax.set_xlabel('Commodity', fontsize=12)
    ax.set_ylabel('Return %', fontsize=12)
    ax.set_title('Average Trading Returns by Commodity', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(commodities)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'returns_by_commodity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_sharpe_by_model(df: pd.DataFrame, output_dir: Path):
    """Plot Sharpe ratios by model."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_order = df.groupby('Model')['Sharpe'].mean().sort_values(ascending=False).index
    
    df_pivot = df.pivot_table(values='Sharpe', index='Model', columns='Commodity', aggfunc='mean')
    df_pivot = df_pivot.reindex(model_order)
    
    df_pivot.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Sharpe Ratio by Model and Commodity', fontsize=14, fontweight='bold')
    ax.legend(title='Commodity', loc='upper right')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'sharpe_by_model.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """Generate heatmap of returns by model and commodity."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    df_pivot = df.pivot_table(values='Return %', index='Model', columns='Commodity', aggfunc='mean')
    
    # Use matplotlib imshow instead of seaborn
    im = ax.imshow(df_pivot.values, cmap='RdYlGn', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return %')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(df_pivot.columns)))
    ax.set_yticks(np.arange(len(df_pivot.index)))
    ax.set_xticklabels(df_pivot.columns)
    ax.set_yticklabels(df_pivot.index)
    
    # Add value annotations
    for i in range(len(df_pivot.index)):
        for j in range(len(df_pivot.columns)):
            ax.text(j, i, f'{df_pivot.iloc[i, j]:.1f}',
                   ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Trading Returns Heatmap (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Commodity', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    output_file = output_dir / 'returns_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_horizon_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot performance comparison across prediction horizons."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Return % by horizon
    df.groupby(['H', 'Commodity'])['Return %'].mean().unstack().plot(
        kind='bar', ax=axes[0], color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
    axes[0].set_title('Return % by Horizon', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Horizon (days)')
    axes[0].set_ylabel('Return %')
    axes[0].legend(title='Commodity')
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Sharpe by horizon
    df.groupby(['H', 'Commodity'])['Sharpe'].mean().unstack().plot(
        kind='bar', ax=axes[1], color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
    axes[1].set_title('Sharpe Ratio by Horizon', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Horizon (days)')
    axes[1].set_ylabel('Sharpe Ratio')
    axes[1].legend(title='Commodity')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'horizon_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_risk_return_scatter(df: pd.DataFrame, output_dir: Path):
    """Generate risk-return scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    commodities = df['Commodity'].unique()
    colors = {'corn': '#2ecc71', 'soybeans': '#3498db', 'wheat': '#e74c3c'}
    markers = {'corn': 'o', 'soybeans': 's', 'wheat': '^'}
    
    for commodity in commodities:
        data = df[df['Commodity'] == commodity]
        ax.scatter(data['Max DD %'], data['Return %'], 
                  c=colors[commodity], marker=markers[commodity],
                  s=100, alpha=0.7, label=commodity.capitalize(), edgecolors='black')
        
        # Add model labels
        for _, row in data.iterrows():
            ax.annotate(f"{row['Model'][:4]}h{row['H']}", 
                       (row['Max DD %'], row['Return %']),
                       fontsize=7, alpha=0.8)
    
    ax.set_xlabel('Max Drawdown %', fontsize=12)
    ax.set_ylabel('Return %', fontsize=12)
    ax.set_title('Risk-Return Profile by Model', fontsize=14, fontweight='bold')
    ax.legend(title='Commodity', loc='upper left')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'risk_return_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def generate_latex_table(df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for paper."""
    # Best model for each commodity-horizon combination
    latex_df = df.groupby(['Commodity', 'H']).apply(
        lambda x: x.nlargest(1, 'Sharpe')[['Model', 'Return %', 'Sharpe', 'Win Rate', 'Profit Factor']]
    ).reset_index(drop=True)
    
    # Format for LaTeX
    latex_content = latex_df.to_latex(index=False, float_format='%.2f')
    
    output_file = output_dir / 'trading_results.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    print(f"Saved: {output_file}")
    
    # Also save full table
    full_latex = df.sort_values(['Commodity', 'H', 'Sharpe'], ascending=[True, True, False]).to_latex(
        index=False, float_format='%.2f')
    
    output_file = output_dir / 'trading_results_full.tex'
    with open(output_file, 'w') as f:
        f.write(full_latex)
    print(f"Saved: {output_file}")


def generate_markdown_report(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive markdown report."""
    report = "# Trading Simulation Results\n\n"
    
    # Summary statistics
    report += "## Summary Statistics\n\n"
    report += f"**Total Models Evaluated:** {len(df)}\n\n"
    report += f"**Models:** {', '.join(sorted(df['Model'].unique()))}\n\n"
    report += f"**Commodities:** {', '.join(sorted(df['Commodity'].unique()))}\n\n"
    report += f"**Horizons:** {sorted(df['H'].unique())}\n\n"
    
    # Overall best
    report += "## Top 10 Overall by Sharpe Ratio\n\n"
    top_sharpe = df.nlargest(10, 'Sharpe')[
        ['Model', 'Commodity', 'H', 'Return %', 'Sharpe', 'Max DD %', 'Win Rate']
    ]
    report += df_to_markdown(top_sharpe)
    report += "\n\n"
    
    # By commodity
    for commodity in sorted(df['Commodity'].unique()):
        report += f"## {commodity.capitalize()} Results\n\n"
        comm_df = df[df['Commodity'] == commodity].sort_values('Sharpe', ascending=False)
        report += df_to_markdown(comm_df[['Model', 'H', 'Return %', 'Sharpe', 'Max DD %', 'Win Rate', 'Profit Factor']])
        report += "\n\n"
        
        # Stats
        report += f"**{commodity.capitalize()} Stats:**\n"
        report += f"- Mean Return: {comm_df['Return %'].mean():.2f}%\n"
        report += f"- Mean Sharpe: {comm_df['Sharpe'].mean():.3f}\n"
        report += f"- Best Model: {comm_df.iloc[0]['Model']} (h{comm_df.iloc[0]['H']})\n\n"
    
    output_file = output_dir / 'trading_report.md'
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate trading graphics and tables')
    parser.add_argument('--csv', type=str, 
                       default='../trading/trading_results/trading_comparison_summary.csv',
                       help='Path to trading comparison CSV')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for generated files')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    
    print(f"\nGenerating tables...")
    generate_summary_by_commodity(df, output_dir)
    generate_summary_by_model(df, output_dir)
    generate_best_performers(df, output_dir)
    
    print(f"\nGenerating visualizations...")
    plot_returns_by_commodity(df, output_dir)
    plot_sharpe_by_model(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_horizon_comparison(df, output_dir)
    plot_risk_return_scatter(df, output_dir)
    
    print(f"\nGenerating LaTeX tables...")
    generate_latex_table(df, output_dir)
    
    print(f"\nGenerating markdown report...")
    generate_markdown_report(df, output_dir)
    
    print(f"\n{'='*60}")
    print("All files generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
