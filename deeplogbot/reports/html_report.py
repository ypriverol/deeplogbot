"""HTML report generator for bot detection results.

Generates an interactive HTML report with:
- Summary statistics
- Classification breakdown
- Interactive charts (embedded)
- Feature analysis
- Top locations tables
"""

import os
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd

from ..utils import logger


class HTMLReportGenerator:
    """Generate interactive HTML reports for bot detection analysis."""

    def __init__(self, output_dir: str):
        """
        Initialize HTML report generator.

        Args:
            output_dir: Directory to save report
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _embed_image(self, image_path: str) -> str:
        """Convert image file to base64 for embedding in HTML."""
        if not os.path.exists(image_path):
            return ""
        try:
            with open(image_path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            return f'data:image/png;base64,{data}'
        except Exception as e:
            logger.warning(f"Failed to embed image {image_path}: {e}")
            return ""

    def _format_number(self, n: float) -> str:
        """Format number for display."""
        if n >= 1e9:
            return f'{n/1e9:.1f}B'
        elif n >= 1e6:
            return f'{n/1e6:.1f}M'
        elif n >= 1e3:
            return f'{n/1e3:.1f}K'
        else:
            return f'{n:,.0f}'

    def _generate_css(self) -> str:
        """Generate CSS styles for the report."""
        return """
        <style>
            :root {
                --primary: #3498db;
                --success: #27ae60;
                --warning: #f39c12;
                --danger: #e74c3c;
                --dark: #2c3e50;
                --light: #ecf0f1;
                --gray: #95a5a6;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                line-height: 1.6;
                color: var(--dark);
                background: #f5f6fa;
                padding: 20px;
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
            }

            header {
                background: linear-gradient(135deg, var(--dark), #34495e);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }

            header h1 {
                font-size: 2em;
                margin-bottom: 10px;
            }

            header .meta {
                opacity: 0.8;
                font-size: 0.9em;
            }

            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }

            .card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            .card.bot { border-left: 4px solid var(--danger); }
            .card.hub { border-left: 4px solid var(--primary); }
            .card.organic { border-left: 4px solid var(--success); }
            .card.automated { border-left: 4px solid var(--warning); }

            .card h3 {
                color: var(--gray);
                font-size: 0.85em;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 10px;
            }

            .card .value {
                font-size: 2em;
                font-weight: bold;
                color: var(--dark);
            }

            .card .subtext {
                font-size: 0.85em;
                color: var(--gray);
                margin-top: 5px;
            }

            section {
                background: white;
                border-radius: 10px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            section h2 {
                color: var(--dark);
                border-bottom: 2px solid var(--light);
                padding-bottom: 10px;
                margin-bottom: 20px;
            }

            .plot-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
            }

            .plot-container {
                background: var(--light);
                border-radius: 8px;
                padding: 15px;
            }

            .plot-container img {
                width: 100%;
                height: auto;
                border-radius: 4px;
            }

            .plot-container h4 {
                color: var(--dark);
                margin-bottom: 10px;
                font-size: 1em;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }

            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid var(--light);
            }

            th {
                background: var(--light);
                font-weight: 600;
                color: var(--dark);
            }

            tr:hover {
                background: #fafbfc;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }

            .stat-box {
                background: var(--light);
                border-radius: 8px;
                padding: 15px;
            }

            .stat-box h4 {
                color: var(--dark);
                margin-bottom: 10px;
            }

            .stat-row {
                display: flex;
                justify-content: space-between;
                padding: 5px 0;
                border-bottom: 1px solid white;
            }

            .stat-row:last-child {
                border-bottom: none;
            }

            .badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: 600;
            }

            .badge-bot { background: #fce4ec; color: var(--danger); }
            .badge-hub { background: #e3f2fd; color: var(--primary); }
            .badge-organic { background: #e8f5e9; color: var(--success); }
            .badge-automated { background: #fff3e0; color: var(--warning); }

            .progress-bar {
                height: 8px;
                background: var(--light);
                border-radius: 4px;
                overflow: hidden;
                margin-top: 5px;
            }

            .progress-bar .fill {
                height: 100%;
                border-radius: 4px;
            }

            .fill-bot { background: var(--danger); }
            .fill-hub { background: var(--primary); }
            .fill-organic { background: var(--success); }

            footer {
                text-align: center;
                padding: 20px;
                color: var(--gray);
                font-size: 0.85em;
            }

            @media (max-width: 768px) {
                .plot-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """

    def _generate_summary_cards(self, stats: Dict[str, Any],
                                classification_method: str) -> str:
        """Generate summary cards HTML."""
        classification = stats.get('classification', {})
        downloads = stats.get('downloads', {})

        cards = []

        # Total locations
        cards.append(f"""
        <div class="card">
            <h3>Total Locations</h3>
            <div class="value">{self._format_number(classification.get('total_locations', 0))}</div>
            <div class="subtext">Analyzed in this run</div>
        </div>
        """)

        # Bot locations
        bot_count = classification.get('bot_locations', 0)
        bot_pct = classification.get('bot_percentage', 0)
        cards.append(f"""
        <div class="card bot">
            <h3>Bot Locations</h3>
            <div class="value">{self._format_number(bot_count)}</div>
            <div class="subtext">{bot_pct}% of all locations</div>
            <div class="progress-bar"><div class="fill fill-bot" style="width: {min(bot_pct, 100)}%"></div></div>
        </div>
        """)

        # Hub locations
        hub_count = classification.get('hub_locations', 0)
        hub_pct = classification.get('hub_percentage', 0)
        cards.append(f"""
        <div class="card hub">
            <h3>Download Hubs</h3>
            <div class="value">{self._format_number(hub_count)}</div>
            <div class="subtext">{hub_pct}% of all locations</div>
            <div class="progress-bar"><div class="fill fill-hub" style="width: {min(hub_pct, 100)}%"></div></div>
        </div>
        """)

        # Total downloads
        total_dl = downloads.get('total_downloads', 0)
        cards.append(f"""
        <div class="card">
            <h3>Total Downloads</h3>
            <div class="value">{self._format_number(total_dl)}</div>
            <div class="subtext">Across all locations</div>
        </div>
        """)

        # Hierarchical cards for deep method
        if classification_method == 'deep' and 'hierarchical' in stats:
            hier = stats['hierarchical']
            if 'behavior_type' in hier:
                organic = hier['behavior_type']['counts'].get('organic', 0)
                automated = hier['behavior_type']['counts'].get('automated', 0)
                total = organic + automated

                cards.append(f"""
                <div class="card organic">
                    <h3>Organic (Human-like)</h3>
                    <div class="value">{self._format_number(organic)}</div>
                    <div class="subtext">{round(organic/total*100, 1) if total > 0 else 0}% of locations</div>
                    <div class="progress-bar"><div class="fill fill-organic" style="width: {organic/total*100 if total > 0 else 0}%"></div></div>
                </div>
                """)

                cards.append(f"""
                <div class="card automated">
                    <h3>Automated</h3>
                    <div class="value">{self._format_number(automated)}</div>
                    <div class="subtext">{round(automated/total*100, 1) if total > 0 else 0}% of locations</div>
                </div>
                """)

        return '<div class="cards">' + ''.join(cards) + '</div>'

    def _generate_plots_section(self, plot_paths: List[str]) -> str:
        """Generate plots section HTML."""
        if not plot_paths:
            return ""

        plot_html = []
        plot_titles = {
            'classification_distribution': 'Classification Distribution',
            'downloads_by_category': 'Downloads by Category',
            'yearly_trends': 'Yearly Trends',
            'feature_distributions': 'Feature Distributions',
            'feature_importance': 'Feature Importance',
            'correlation_matrix': 'Feature Correlations',
            'category_feature_comparison': 'Features by Category',
            'geographic_distribution': 'Geographic Distribution',
            'temporal_patterns': 'Temporal Patterns',
            'anomaly_analysis': 'Anomaly Analysis',
        }

        for path in plot_paths:
            name = Path(path).stem
            title = plot_titles.get(name, name.replace('_', ' ').title())
            embedded = self._embed_image(path)
            if embedded:
                plot_html.append(f"""
                <div class="plot-container">
                    <h4>{title}</h4>
                    <img src="{embedded}" alt="{title}">
                </div>
                """)

        if not plot_html:
            return ""

        return f"""
        <section>
            <h2>Visualizations</h2>
            <div class="plot-grid">
                {''.join(plot_html)}
            </div>
        </section>
        """

    def _generate_feature_stats_section(self, stats: Dict[str, Any]) -> str:
        """Generate feature statistics section."""
        features = stats.get('features', {})
        if not features:
            return ""

        boxes = []
        for feature, feature_stats in list(features.items())[:8]:  # Top 8 features
            rows = []
            for key, value in feature_stats.items():
                if key in ['mean', 'std', 'median', 'min', 'max']:
                    label = key.title()
                    rows.append(f'<div class="stat-row"><span>{label}</span><span>{value:,.4g}</span></div>')

            boxes.append(f"""
            <div class="stat-box">
                <h4>{feature.replace('_', ' ').title()}</h4>
                {''.join(rows)}
            </div>
            """)

        return f"""
        <section>
            <h2>Feature Statistics</h2>
            <div class="stats-grid">
                {''.join(boxes)}
            </div>
        </section>
        """

    def _generate_top_locations_table(self, df: pd.DataFrame, category: str,
                                      top_n: int = 20) -> str:
        """Generate table of top locations for a category."""
        # Use hierarchical classification columns
        hub_subcategories = {'mirror', 'institutional_hub', 'data_aggregator'}
        if category == 'bot' and 'automation_category' in df.columns:
            subset = df[df['automation_category'] == 'bot'].sort_values('unique_users', ascending=False).head(top_n)
            badge_class = 'badge-bot'
        elif category == 'hub' and 'subcategory' in df.columns:
            subset = df[df['subcategory'].isin(hub_subcategories)].sort_values('downloads_per_user', ascending=False).head(top_n)
            badge_class = 'badge-hub'
        else:
            return ""

        if len(subset) == 0:
            return ""

        rows = []
        for _, row in subset.iterrows():
            country = str(row.get('country', 'N/A'))[:20]
            city = str(row.get('city', ''))[:20] if pd.notna(row.get('city')) else ''
            users = int(row.get('unique_users', 0))
            dpu = row.get('downloads_per_user', 0)
            total = int(row.get('total_downloads', 0))

            rows.append(f"""
            <tr>
                <td>{country}</td>
                <td>{city}</td>
                <td>{users:,}</td>
                <td>{dpu:,.1f}</td>
                <td>{self._format_number(total)}</td>
            </tr>
            """)

        title = 'Bot Locations' if category == 'bot' else 'Download Hub Locations'

        return f"""
        <section>
            <h2><span class="badge {badge_class}">{category.upper()}</span> Top {title}</h2>
            <table>
                <thead>
                    <tr>
                        <th>Country</th>
                        <th>City</th>
                        <th>Users</th>
                        <th>DL/User</th>
                        <th>Total Downloads</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </section>
        """

    def _generate_hierarchical_section(self, stats: Dict[str, Any]) -> str:
        """Generate hierarchical classification breakdown section."""
        hier = stats.get('hierarchical', {})
        if not hier:
            return ""

        html_parts = ['<section><h2>Hierarchical Classification Breakdown</h2>']

        # Level 1: Behavior Type
        if 'behavior_type' in hier:
            bt = hier['behavior_type']
            html_parts.append('<h3>Level 1: Behavior Type</h3>')
            html_parts.append('<div class="stats-grid">')
            for category, count in bt['counts'].items():
                pct = bt['percentages'].get(category, 0)
                badge_class = f'badge-{category}'
                html_parts.append(f"""
                <div class="stat-box">
                    <h4><span class="badge {badge_class}">{category.upper()}</span></h4>
                    <div class="stat-row"><span>Locations</span><span>{count:,}</span></div>
                    <div class="stat-row"><span>Percentage</span><span>{pct}%</span></div>
                </div>
                """)
            html_parts.append('</div>')

        # Level 2: Automation Category
        if 'automation_category' in hier:
            ac = hier['automation_category']
            html_parts.append('<h3>Level 2: Automation Category (within Automated)</h3>')
            html_parts.append('<div class="stats-grid">')
            for category, count in ac['counts'].items():
                pct = ac['percentages'].get(category, 0)
                badge_class = 'badge-bot' if category == 'bot' else 'badge-hub'
                html_parts.append(f"""
                <div class="stat-box">
                    <h4><span class="badge {badge_class}">{category.upper().replace('_', ' ')}</span></h4>
                    <div class="stat-row"><span>Locations</span><span>{count:,}</span></div>
                    <div class="stat-row"><span>% of Automated</span><span>{pct}%</span></div>
                </div>
                """)
            html_parts.append('</div>')

        # Level 3: Subcategories
        if 'subcategory' in hier and 'by_parent' in hier['subcategory']:
            html_parts.append('<h3>Level 3: Subcategories</h3>')
            for parent, subcats in hier['subcategory']['by_parent'].items():
                if subcats:
                    html_parts.append(f'<h4 style="margin-top: 15px;">{parent.upper().replace("_", " ")}</h4>')
                    html_parts.append('<table><thead><tr><th>Subcategory</th><th>Count</th></tr></thead><tbody>')
                    for subcat, count in subcats.items():
                        html_parts.append(f'<tr><td>{subcat}</td><td>{count:,}</td></tr>')
                    html_parts.append('</tbody></table>')

        html_parts.append('</section>')
        return ''.join(html_parts)

    def generate(self, df: pd.DataFrame, stats: Dict[str, Any],
                 classification_method: str = 'rules',
                 plot_paths: Optional[List[str]] = None,
                 feature_importance: Optional[Dict[str, float]] = None) -> str:
        """
        Generate the complete HTML report.

        Args:
            df: Analysis DataFrame
            stats: Statistics dictionary from StatisticsCalculator
            classification_method: 'rules' or 'deep'
            plot_paths: Optional list of paths to plot images
            feature_importance: Optional feature importance dict (reserved for future use)

        Returns:
            Path to generated HTML file
        """
        # Note: feature_importance parameter reserved for future feature importance section
        _ = feature_importance  # Acknowledge unused parameter
        report_path = self.output_dir / 'report.html'

        # Build the report
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>Bot Detection Report</title>',
            self._generate_css(),
            '</head>',
            '<body>',
            '<div class="container">',
        ]

        # Header
        method_display = 'Deep Architecture (Hierarchical)' if classification_method == 'deep' else 'Rule-based'
        html_parts.append(f"""
        <header>
            <h1>Bot Detection Report</h1>
            <div class="meta">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
                Method: {method_display} |
                Locations: {stats.get('classification', {}).get('total_locations', 0):,}
            </div>
        </header>
        """)

        # Summary cards
        html_parts.append(self._generate_summary_cards(stats, classification_method))

        # Hierarchical breakdown (deep method)
        if classification_method == 'deep':
            html_parts.append(self._generate_hierarchical_section(stats))

        # Plots
        if plot_paths:
            html_parts.append(self._generate_plots_section(plot_paths))

        # Feature statistics
        html_parts.append(self._generate_feature_stats_section(stats))

        # Top locations tables
        html_parts.append(self._generate_top_locations_table(df, 'bot'))
        html_parts.append(self._generate_top_locations_table(df, 'hub'))

        # Footer
        html_parts.append("""
        <footer>
            <p>Generated by DeepLogBot | Bot and Download Hub Detection System</p>
        </footer>
        """)

        html_parts.extend(['</div>', '</body>', '</html>'])

        # Write the file
        html_content = '\n'.join(html_parts)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to: {report_path}")
        return str(report_path)
