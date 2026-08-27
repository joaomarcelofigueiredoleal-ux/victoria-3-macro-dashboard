# 🌍 Victoria 3 Macroeconomic Dashboard

An interactive, analytical web application built with Streamlit and Plotly that brings rigorous macroeconomic tracking to Victoria 3 campaigns. 

This dashboard transforms raw game data into a professional-grade economic dossier, allowing players to analyze their nation's historical performance, track structural growth, and run head-to-head convergence analyses against rival hegemonies.

## 📊 Core Analytical Features

*   **Global Landscape Meta-Analysis:** Cross-compare the GDP, GDP per capita, Population, and Standard of Living (SoL) trajectories of multiple Great Powers simultaneously.
*   **Structural Growth Accounting:** Break down annual and trending economic growth into Intensive (Productivity/GDP per capita) and Extensive (Demographic/Population) components.
*   **Macroeconomic Volatility & Crises:** Map periods of economic expansion versus contraction, track major drawdowns, and measure overall economic stability using rolling standard deviations.
*   **Bilateral Convergence Dynamics:** Run head-to-head comparisons to track catch-up velocities and productivity parity between a target economy and a global benchmark.
*   **Cross-Campaign Meta-Analysis:** Load multiple `.csv` campaign files to objectively compare the economic outcomes of entirely different playthroughs on the same timeline.

## ⚙️ How to Use

1. Export your Victoria 3 campaign data to `.csv` format.
2. Place the `.csv` files in the root directory of this project.
3. Select your save file from the dropdown menu in the dashboard to instantly generate the analytical dossiers.

**Built with:** Python, Streamlit, Plotly, Pandas, and SciPy.
