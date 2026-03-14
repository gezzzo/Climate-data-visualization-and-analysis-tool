from flask import Flask, render_template, request, jsonify
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Get available cities from content folder
CONTENT_PATH = 'content/'

def get_available_cities():
    """Get list of available cities from .nc files"""
    cities = []
    for f in os.listdir(CONTENT_PATH):
        if f.endswith('.nc'):
            city_name = f.replace('.nc', '')
            cities.append(city_name)
    return sorted(cities)

def load_city_data(city_name):
    """Load and preprocess city climate data"""
    file_path = os.path.join(CONTENT_PATH, f'{city_name}.nc')
    ds = xr.open_dataset(file_path)
    
    # Extract temperature and convert to Celsius
    t2m = ds['t2m'].squeeze() - 273.15
    
    # Remove leap day (Feb 29)
    t2m = t2m.sel(time=~((t2m.time.dt.month == 2) & (t2m.time.dt.day == 29)))
    
    return t2m

def get_30_year_data(t2m):
    """Get last 30 years of data"""
    end_time = pd.to_datetime(str(t2m.time.values[-1]))
    start_time = end_time - pd.DateOffset(years=30)
    return t2m.sel(time=slice(np.datetime64(start_time), np.datetime64(end_time)))

def get_trend(t2m):
    """Calculate linear trend"""
    time = t2m.time
    x = (time - time[0]) / np.timedelta64(1, 'D')
    y = t2m.values.flatten()
    
    mask = ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
    trend = slope * x + intercept
    
    return time, trend, slope, r_value**2

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

def deseasonalize_and_detrend(t2m):
    """Remove seasonal cycle and linear trend"""
    climatology = t2m.groupby("time.dayofyear").mean("time")
    deseasonalized = t2m.groupby("time.dayofyear") - climatology
    
    times = deseasonalized.time.values.astype('datetime64[h]').astype(float)
    slope, intercept, _, _, _ = linregress(times, deseasonalized.values)
    trend = slope * times + intercept
    detrended = deseasonalized - trend
    
    return detrended

@app.route('/')
def index():
    """Main page"""
    cities = get_available_cities()
    return render_template('index.html', cities=cities)

@app.route('/compare', methods=['POST'])
def compare():
    """Compare two cities"""
    data = request.json
    city1 = data.get('city1')
    city2 = data.get('city2')
    
    if not city1 or not city2:
        return jsonify({'error': 'Please select two cities'}), 400
    
    if city1 == city2:
        return jsonify({'error': 'Please select two different cities'}), 400
    
    try:
        # Load data
        t2m_1 = load_city_data(city1)
        t2m_2 = load_city_data(city2)
        
        # Get 30-year data
        t2m_1_30y = get_30_year_data(t2m_1)
        t2m_2_30y = get_30_year_data(t2m_2)
        
        results = {}
        
        # === 1. Basic Statistics ===
        stats = calculate_basic_stats(t2m_1_30y, t2m_2_30y, city1, city2)
        results['basic_stats'] = stats
        
        # === 2. Temperature Trend Analysis ===
        trend_result = create_trend_analysis(t2m_1_30y, t2m_2_30y, city1, city2)
        results['trend'] = trend_result
        
        # === 3. Long-term Trend ===
        longterm_result = create_longterm_trend(t2m_1, t2m_2, city1, city2)
        results['longterm_trend'] = longterm_result
        
        # === 4. Temperature Difference ===
        diff_result = create_temperature_difference(t2m_1_30y, t2m_2_30y, city1, city2)
        results['difference'] = diff_result
        
        # === 5. Monthly Climatology ===
        monthly_result = create_monthly_climatology(t2m_1_30y, t2m_2_30y, city1, city2)
        results['monthly'] = monthly_result
        
        # === 6. Variability Analysis ===
        variability_result = create_variability_analysis(t2m_1_30y, t2m_2_30y, city1, city2)
        results['variability'] = variability_result
        
        # === 7. Anomaly Distribution ===
        anomaly_result = create_anomaly_distribution(t2m_1_30y, t2m_2_30y, city1, city2)
        results['anomaly'] = anomaly_result
        
        # === 8. Monthly Variability ===
        monthly_var_result = create_monthly_variability(t2m_1_30y, t2m_2_30y, city1, city2)
        results['monthly_variability'] = monthly_var_result
        
        # === 9. Frequency Domain Analysis ===
        freq_result = create_frequency_analysis(t2m_1_30y, t2m_2_30y, city1, city2)
        results['frequency'] = freq_result
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def to_python_float(val):
    """Convert numpy types to Python native types for JSON serialization"""
    if hasattr(val, 'item'):
        return val.item()
    return float(val)

def calculate_basic_stats(t2m_1, t2m_2, city1, city2):
    """Calculate basic statistics for both cities"""
    return {
        'city1': {
            'name': city1,
            'mean': round(to_python_float(t2m_1.mean()), 2),
            'max': round(to_python_float(t2m_1.max()), 2),
            'min': round(to_python_float(t2m_1.min()), 2),
            'std': round(to_python_float(t2m_1.std()), 2)
        },
        'city2': {
            'name': city2,
            'mean': round(to_python_float(t2m_2.mean()), 2),
            'max': round(to_python_float(t2m_2.max()), 2),
            'min': round(to_python_float(t2m_2.min()), 2),
            'std': round(to_python_float(t2m_2.std()), 2)
        },
        'explanation': {
            'mean': 'Average temperature over the 30-year period. Higher values indicate warmer climates.',
            'max': 'Maximum recorded temperature. Shows extreme heat potential.',
            'min': 'Minimum recorded temperature. Shows extreme cold potential.',
            'std': 'Standard deviation measures how much temperatures vary from the average. Higher values mean more variable climate.'
        }
    }

def create_trend_analysis(t2m_1, t2m_2, city1, city2):
    """Create temperature trend analysis"""
    time1, trend1, slope1, r2_1 = get_trend(t2m_1)
    time2, trend2, slope2, r2_2 = get_trend(t2m_2)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(t2m_1.time, t2m_1.values.flatten(), alpha=0.3, label=f'{city1}', color='#e74c3c')
    ax.plot(time1, trend1, '--', label=f'{city1} Trend ({slope1*365:.3f} °C/year)', color='#c0392b', linewidth=2)
    
    ax.plot(t2m_2.time, t2m_2.values.flatten(), alpha=0.3, label=f'{city2}', color='#3498db')
    ax.plot(time2, trend2, '--', label=f'{city2} Trend ({slope2*365:.3f} °C/year)', color='#2980b9', linewidth=2)
    
    ax.set_title('Temperature and Linear Trend (Last 30 Years)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature [°C]', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    img_str = fig_to_base64(fig)
    
    # Determine which city is warming faster
    faster_warming = city1 if slope1 > slope2 else city2
    slower_warming = city2 if slope1 > slope2 else city1
    diff = abs(slope1 - slope2) * 365
    
    return {
        'image': img_str,
        'data': {
            'city1_slope': round(to_python_float(slope1 * 365), 4),
            'city2_slope': round(to_python_float(slope2 * 365), 4),
            'city1_r2': round(to_python_float(r2_1), 4),
            'city2_r2': round(to_python_float(r2_2), 4)
        },
        'explanation': f"""
        <strong>What does this show?</strong><br>
        This graph shows hourly temperature data with a linear trend line for the last 30 years.<br><br>
        
        <strong>Key Findings:</strong><br>
        • <strong>{city1}</strong> is warming at <strong>{slope1*365:.3f} °C per year</strong><br>
        • <strong>{city2}</strong> is warming at <strong>{slope2*365:.3f} °C per year</strong><br>
        • <strong>{faster_warming}</strong> is warming {diff:.3f} °C/year faster than {slower_warming}<br><br>
        
        <strong>Why does this matter?</strong><br>
        Climate change affects different regions differently. A positive slope indicates warming over time. 
        Even small differences in warming rates can lead to significant temperature changes over decades.
        """
    }

def create_longterm_trend(t2m_1, t2m_2, city1, city2):
    """Create long-term trend comparison (full dataset)"""
    time1, trend1, slope1, r2_1 = get_trend(t2m_1)
    time2, trend2, slope2, r2_2 = get_trend(t2m_2)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(time1, trend1, label=f'{city1} Trend ({slope1*365:.3f} °C/year)', color='#e74c3c', linewidth=2)
    ax.plot(time2, trend2, label=f'{city2} Trend ({slope2*365:.3f} °C/year)', color='#3498db', linewidth=2)
    
    ax.set_title('Long-term Temperature Trends (Full Historical Data)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Trend Temperature [°C]', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    img_str = fig_to_base64(fig)
    
    return {
        'image': img_str,
        'data': {
            'city1_slope': round(to_python_float(slope1 * 365), 4),
            'city2_slope': round(to_python_float(slope2 * 365), 4)
        },
        'explanation': f"""
        <strong>What does this show?</strong><br>
        This shows the linear temperature trend over the entire available historical period (typically from 1940s-1950s).<br><br>
        
        <strong>Key Findings:</strong><br>
        • Long-term warming rate for <strong>{city1}</strong>: {slope1*365:.3f} °C/year<br>
        • Long-term warming rate for <strong>{city2}</strong>: {slope2*365:.3f} °C/year<br><br>
        
        <strong>Why does this matter?</strong><br>
        Long-term trends help us understand climate change patterns over multiple decades. 
        Comparing recent (30-year) vs long-term trends can reveal if warming is accelerating.
        """
    }

def create_temperature_difference(t2m_1, t2m_2, city1, city2):
    """Create temperature difference analysis"""
    # Running means
    t2m_1_rm_30 = t2m_1.rolling(time=24*30, center=True).mean()
    t2m_2_rm_30 = t2m_2.rolling(time=24*30, center=True).mean()
    diff_rm_30 = t2m_1_rm_30 - t2m_2_rm_30
    
    t2m_1_rm_365 = t2m_1.rolling(time=24*365, center=True).mean()
    t2m_2_rm_365 = t2m_2.rolling(time=24*365, center=True).mean()
    diff_rm_365 = t2m_1_rm_365 - t2m_2_rm_365
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(diff_rm_30.time, diff_rm_30, label='30-Day Running Mean Difference', color='#9b59b6', alpha=0.7)
    ax.plot(diff_rm_365.time, diff_rm_365, label='365-Day Running Mean Difference', color='#e67e22', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title(f'Temperature Difference: {city1} minus {city2}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature Difference [°C]', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    img_str = fig_to_base64(fig)
    
    avg_diff = float(diff_rm_365.mean())
    warmer_city = city1 if avg_diff > 0 else city2
    
    return {
        'image': img_str,
        'data': {
            'avg_difference': round(to_python_float(avg_diff), 2),
            'max_difference': round(to_python_float(diff_rm_365.max()), 2),
            'min_difference': round(to_python_float(diff_rm_365.min()), 2)
        },
        'explanation': f"""
        <strong>What does this show?</strong><br>
        This graph shows the temperature difference between the two cities over time.<br>
        Positive values mean {city1} is warmer; negative values mean {city2} is warmer.<br><br>
        
        <strong>Key Findings:</strong><br>
        • Average difference: <strong>{avg_diff:.2f} °C</strong> ({warmer_city} is generally warmer)<br>
        • The purple line (30-day average) shows seasonal variations<br>
        • The orange line (yearly average) shows long-term patterns<br><br>
        
        <strong>Why does this matter?</strong><br>
        Understanding temperature differences helps with planning relocations, comparing living conditions, 
        and understanding regional climate patterns. Seasonal differences can be more extreme than annual averages suggest.
        """
    }

def create_monthly_climatology(t2m_1, t2m_2, city1, city2):
    """Create monthly average comparison"""
    t2m_1_monthly = t2m_1.groupby("time.month").mean("time")
    t2m_2_monthly = t2m_2.groupby("time.month").mean("time")
    
    months = np.arange(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    bars1 = ax.bar(months - width/2, t2m_1_monthly, width=width, label=city1, color='#e74c3c')
    bars2 = ax.bar(months + width/2, t2m_2_monthly, width=width, label=city2, color='#3498db')
    
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Temperature [°C]', fontsize=12)
    ax.set_title('Monthly Average Temperatures', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    img_str = fig_to_base64(fig)
    
    # Find hottest and coldest months
    hottest_month_1 = month_names[int(t2m_1_monthly.argmax())]
    coldest_month_1 = month_names[int(t2m_1_monthly.argmin())]
    hottest_month_2 = month_names[int(t2m_2_monthly.argmax())]
    coldest_month_2 = month_names[int(t2m_2_monthly.argmin())]
    
    summer_diff = float(t2m_1_monthly.sel(month=7) - t2m_2_monthly.sel(month=7))
    winter_diff = float(t2m_1_monthly.sel(month=1) - t2m_2_monthly.sel(month=1))
    
    return {
        'image': img_str,
        'data': {
            'city1_monthly': [round(to_python_float(v), 1) for v in t2m_1_monthly.values],
            'city2_monthly': [round(to_python_float(v), 1) for v in t2m_2_monthly.values]
        },
        'explanation': f"""
        <strong>What does this show?</strong><br>
        Monthly average temperatures based on 30 years of data. This is called the "climatology" - 
        the expected temperature for each month.<br><br>
        
        <strong>Key Findings:</strong><br>
        • <strong>{city1}</strong>: Hottest in {hottest_month_1}, coldest in {coldest_month_1}<br>
        • <strong>{city2}</strong>: Hottest in {hottest_month_2}, coldest in {coldest_month_2}<br>
        • Summer (July) difference: {summer_diff:.1f} °C<br>
        • Winter (January) difference: {winter_diff:.1f} °C<br><br>
        
        <strong>Why does this matter?</strong><br>
        Monthly climatology helps understand seasonal patterns. Mediterranean cities typically have milder winters 
        and hot summers, while continental cities have larger temperature swings between seasons.
        """
    }

def create_variability_analysis(t2m_1, t2m_2, city1, city2):
    """Create variability analysis at different time scales"""
    # Calculate standard deviations at different scales
    std_hourly_1 = float(t2m_1.std())
    std_hourly_2 = float(t2m_2.std())
    
    daily_1 = t2m_1.resample(time='1D').mean()
    daily_2 = t2m_2.resample(time='1D').mean()
    std_daily_1 = float(daily_1.std())
    std_daily_2 = float(daily_2.std())
    
    monthly_1 = t2m_1.resample(time='1ME').mean()
    monthly_2 = t2m_2.resample(time='1ME').mean()
    std_monthly_1 = float(monthly_1.std())
    std_monthly_2 = float(monthly_2.std())
    
    yearly_1 = t2m_1.resample(time='1YE').mean()
    yearly_2 = t2m_2.resample(time='1YE').mean()
    std_yearly_1 = float(yearly_1.std())
    std_yearly_2 = float(yearly_2.std())
    
    labels = ['Hourly', 'Daily', 'Monthly', 'Yearly']
    city1_std = [std_hourly_1, std_daily_1, std_monthly_1, std_yearly_1]
    city2_std = [std_hourly_2, std_daily_2, std_monthly_2, std_yearly_2]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(x - width/2, city1_std, width, label=city1, color='#e74c3c')
    ax.bar(x + width/2, city2_std, width, label=city2, color='#3498db')
    
    ax.set_ylabel('Standard Deviation [°C]', fontsize=12)
    ax.set_title('Temperature Variability at Different Time Scales', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    
    img_str = fig_to_base64(fig)
    
    more_variable = city1 if std_daily_1 > std_daily_2 else city2
    
    return {
        'image': img_str,
        'data': {
            'city1': {'hourly': round(to_python_float(std_hourly_1), 2), 'daily': round(to_python_float(std_daily_1), 2), 
                     'monthly': round(to_python_float(std_monthly_1), 2), 'yearly': round(to_python_float(std_yearly_1), 2)},
            'city2': {'hourly': round(to_python_float(std_hourly_2), 2), 'daily': round(to_python_float(std_daily_2), 2), 
                     'monthly': round(to_python_float(std_monthly_2), 2), 'yearly': round(to_python_float(std_yearly_2), 2)}
        },
        'explanation': f"""
        <strong>What does this show?</strong><br>
        Standard deviation of temperature at different time scales (hourly to yearly averages).<br><br>
        
        <strong>Key Findings:</strong><br>
        • <strong>{more_variable}</strong> has more variable temperatures overall<br>
        • Hourly variability includes day/night cycles<br>
        • Daily variability shows weather system changes<br>
        • Monthly and yearly variability shows seasonal and climate variations<br><br>
        
        <strong>Why does this matter?</strong><br>
        High variability means less predictable temperatures and potentially more extreme events. 
        Coastal and island cities typically have lower variability due to the moderating effect of the sea.
        Continental cities have higher variability.
        """
    }

def create_anomaly_distribution(t2m_1, t2m_2, city1, city2):
    """Create temperature anomaly distribution"""
    sns.set_style("whitegrid")
    
    def get_anomalies(t2m):
        # Daily anomalies
        t2m_daily_rm = t2m.rolling(time=24, center=True).mean()
        daily_clim = t2m.groupby('time.dayofyear').mean('time')
        daily_anomalies = t2m_daily_rm.groupby('time.dayofyear') - daily_clim
        daily_vals = daily_anomalies.values.flatten()
        daily_vals = daily_vals[~np.isnan(daily_vals)]
        return daily_vals
    
    anomalies_1 = get_anomalies(t2m_1)
    anomalies_2 = get_anomalies(t2m_2)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    sns.kdeplot(anomalies_1, label=city1, linewidth=2, color='#e74c3c', ax=ax)
    sns.kdeplot(anomalies_2, label=city2, linewidth=2, color='#3498db', ax=ax)
    
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title('Distribution of Daily Temperature Anomalies', fontsize=14, fontweight='bold')
    ax.set_xlabel('Temperature Anomaly [°C]', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_xlim(-15, 15)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    img_str = fig_to_base64(fig)
    
    wider_dist = city1 if np.std(anomalies_1) > np.std(anomalies_2) else city2
    
    return {
        'image': img_str,
        'data': {
            'city1_std': round(to_python_float(np.std(anomalies_1)), 2),
            'city2_std': round(to_python_float(np.std(anomalies_2)), 2)
        },
        'explanation': f"""
        <strong>What does this show?</strong><br>
        Distribution of how much daily temperatures deviate from the "normal" (climatological average) for that day.<br><br>
        
        <strong>Key Findings:</strong><br>
        • A narrower curve = more consistent temperatures<br>
        • A wider curve = more temperature extremes<br>
        • <strong>{wider_dist}</strong> has a wider distribution (more variability)<br>
        • {city1} anomaly std: {np.std(anomalies_1):.2f}°C, {city2} anomaly std: {np.std(anomalies_2):.2f}°C<br><br>
        
        <strong>Why does this matter?</strong><br>
        Temperature anomalies affect agriculture, energy demand, and human comfort. 
        Cities with narrower distributions are more predictable and may require less energy for heating/cooling.
        """
    }

def create_monthly_variability(t2m_1, t2m_2, city1, city2):
    """Create monthly variability analysis"""
    detrended_1 = deseasonalize_and_detrend(t2m_1)
    detrended_2 = deseasonalize_and_detrend(t2m_2)
    
    std_monthly_1 = detrended_1.groupby("time.month").std()
    std_monthly_2 = detrended_2.groupby("time.month").std()
    
    months = np.arange(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.bar(std_monthly_1['month'] - 0.2, std_monthly_1, width=0.4, label=city1, color='#e74c3c', alpha=0.8)
    ax.bar(std_monthly_2['month'] + 0.2, std_monthly_2, width=0.4, label=city2, color='#3498db', alpha=0.8)
    
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Standard Deviation [°C]', fontsize=12)
    ax.set_title('Monthly Temperature Variability (Deseasonalized & Detrended)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    img_str = fig_to_base64(fig)
    
    # Find most and least variable months
    most_var_month_1 = month_names[int(std_monthly_1.argmax())]
    least_var_month_1 = month_names[int(std_monthly_1.argmin())]
    
    return {
        'image': img_str,
        'data': {
            'city1_monthly_std': [round(to_python_float(v), 2) for v in std_monthly_1.values],
            'city2_monthly_std': [round(to_python_float(v), 2) for v in std_monthly_2.values]
        },
        'explanation': f"""
        <strong>What does this show?</strong><br>
        Temperature variability for each month, after removing seasonal patterns and long-term trends.<br><br>
        
        <strong>Key Findings:</strong><br>
        • Winter months typically show higher variability (more weather systems)<br>
        • Summer months are usually more stable<br>
        • Coastal cities tend to have lower monthly variability than continental cities<br><br>
        
        <strong>Why does this matter?</strong><br>
        Understanding when temperatures are most unpredictable helps with planning activities, 
        agricultural decisions, and understanding climate risks at different times of year.
        """
    }

def create_frequency_analysis(t2m_1, t2m_2, city1, city2):
    """Create frequency domain analysis"""
    t2m_daily_1 = t2m_1.resample(time='1D').mean()
    t2m_daily_2 = t2m_2.resample(time='1D').mean()
    
    ts1 = t2m_daily_1.values.flatten()
    ts2 = t2m_daily_2.values.flatten()
    
    ts1 = ts1[~np.isnan(ts1)]
    ts2 = ts2[~np.isnan(ts2)]
    
    fs = 1  # 1/day
    
    f1, Pxx1 = welch(ts1, fs=fs, nperseg=min(1856, len(ts1)//2))
    f2, Pxx2 = welch(ts2, fs=fs, nperseg=min(1856, len(ts2)//2))
    
    period1 = 1 / f1
    period2 = 1 / f2
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(period1, Pxx1, label=city1, color='#e74c3c', linewidth=2)
    ax.plot(period2, Pxx2, label=city2, color='#3498db', linewidth=2)
    
    ax.set_xlabel('Period (days)', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title('Frequency Domain Analysis (Power Spectrum)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(2, 2000)
    
    ax.set_xticks([2, 5, 10, 30, 90, 180, 365, 730, 1000, 1500])
    ax.set_xticklabels(['2d', '5d', '10d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '4y'])
    
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    ax.invert_xaxis()
    
    img_str = fig_to_base64(fig)
    
    return {
        'image': img_str,
        'explanation': f"""
        <strong>What does this show?</strong><br>
        Power spectrum analysis reveals which cycles (periods) have the most influence on temperature.<br><br>
        
        <strong>How to read this chart:</strong><br>
        • Peaks indicate dominant cycles in the temperature data<br>
        • The 365-day (1 year) peak shows the seasonal cycle<br>
        • Shorter periods (days to weeks) show weather system influences<br>
        • Higher power = stronger influence of that cycle<br><br>
        
        <strong>Why does this matter?</strong><br>
        Understanding temperature cycles helps with:
        <ul>
        <li>Weather prediction</li>
        <li>Identifying multi-year climate patterns</li>
        <li>Understanding the relative importance of seasonal vs. weather variations</li>
        </ul>
        """
    }
