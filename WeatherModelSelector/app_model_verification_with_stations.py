"""Weather Model Verification — live forecast comparison, not observed verification.
Run from WeatherModelSelector: streamlit run app_model_verification.py
Requires: streamlit, pandas, numpy, plotly, joblib.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / 'results' / 'step49B_live_31locations_predictions.csv'
HIST = ROOT / 'results' / 'step48_forecast_only_results.csv'
MODEL_DIR = ROOT / 'models'
MODELS = ['ecmwf_ifs', 'gfs_seamless', 'icon_seamless', 'gem_seamless']
NAMES = {'ecmwf_ifs':'ECMWF', 'gfs_seamless':'GFS', 'icon_seamless':'ICON', 'gem_seamless':'GEM'}
COLORS = {'ecmwf_ifs':'#0072B2', 'gfs_seamless':'#E69F00', 'icon_seamless':'#009E73', 'gem_seamless':'#CC79A7'}
PARAMS = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
UNITS = {'temperature':'°C','humidity':'%','precipitation':'mm','wind_speed':'km/h','pressure':'hPa'}
LABELS = {'temperature':'Temperature', 'humidity':'Humidity', 'precipitation':'Precipitation', 'wind_speed':'Wind speed', 'pressure':'Sea-level pressure'}
CITY_REGION = {
 'Amritsar':'North', 'Chandigarh':'North','Dehradun':'North','Delhi':'North','Jaipur':'North','Lucknow':'North','Srinagar':'North',
 'Bhopal':'Central','Indore':'Central','Jabalpur':'Central','Raipur':'Central',
 'Aurangabad/Chhatrapati Sambhajinagar':'West','Karad':'West','Kolhapur':'West','Mumbai':'West','Nagpur':'West','Nashik':'West','Pune':'West','Satara':'West',
 'Bengaluru':'South','Chennai':'South','Coimbatore':'South','Hyderabad':'South','Kochi':'South','Mysuru':'South','Thiruvananthapuram':'South',
 'Bhubaneswar':'East','Guwahati':'East','Kolkata':'East','Patna':'East','Ranchi':'East'
}

st.set_page_config(page_title='Model Verification | Weather', page_icon='🌤️', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
:root {color-scheme:dark}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse at 78% -12%,#1c4982 0%,#0d2445 37%,#091a32 77%)!important; color:#eaf5ff!important;}
[data-testid="stHeader"] {background:rgba(9,26,50,.86)!important;}
[data-testid="stToolbar"], [data-testid="stDecoration"] {background:transparent!important;}
.block-container {padding-top:1.3rem;max-width:1540px;}
h1,h2,h3,h4,p,li {color:#eef6ff!important;}
h1,h2,h3 {letter-spacing:-.025em;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#a9c8e6!important;}
.hero {background:linear-gradient(115deg,#164e90,#2872bc 57%,#174276);border:1px solid #62a4f450;border-radius:21px;padding:26px 30px;color:#fff;margin:15px 0 18px;box-shadow:0 12px 35px #040d1c65;}
.hero .eyebrow {font-size:12px;letter-spacing:.15em;opacity:.84;font-weight:700;}
.hero h2 {font-size:35px;color:white!important;margin:5px 0;}
.hero p {color:#e5f3ff!important;margin:5px 0;}
.geo-row {display:flex;flex-wrap:wrap;gap:12px;margin:17px 0 15px;}
.geo-chip {display:flex;gap:10px;align-items:center;background:#092c55a8;border:1px solid #96d7ff80;border-radius:14px;padding:11px 16px;box-shadow:inset 0 1px #ffffff17;}
.geo-chip .geo-label {display:block;color:#b6d8f8;font-size:12px;font-weight:650;letter-spacing:.035em;}
.geo-chip .geo-value {display:block;color:#fff;font-size:22px;line-height:1.25;font-weight:800;font-variant-numeric:tabular-nums;}
.geo-title {font-size:19px;font-weight:750;color:#fff;margin-top:12px;}
.meta {font-size:12px;color:#bad7ef;}
.section {font-size:18px;font-weight:750;margin-top:22px;margin-bottom:7px;color:#e7f5ff;}
.pill {display:inline-block;padding:4px 10px;border-radius:30px;background:#1e5182;color:#d9f2ff;font-size:12px;font-weight:700;border:1px solid #4b9be270;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#16395f,#102b4d)!important;padding:14px 18px;border:1px solid #3472a970;border-radius:15px;min-height:112px;box-shadow:0 4px 16px #05132738;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {color:#b3d0e9!important;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {color:#f4faff!important;font-size:26px;}
[data-testid="stPlotlyChart"] {background:#112c4c!important;border:1px solid #3674a980;border-radius:16px;padding:6px;}
[data-testid="stDataFrame"] {border:1px solid #3674a980;border-radius:12px;background:#122c4a;}
[data-testid="stSelectbox"] label, [data-testid="stWidgetLabel"] p {color:#cde4ff!important;}
[data-baseweb="select"] > div {background:#16375a!important;border-color:#457ba9!important;color:#eff7ff!important;}
/* Popover options are portalled outside the select widget. Apply text color to EVERY descendant. */
[data-baseweb="popover"], [data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {
  background:#14395f!important; color:#ffffff!important;
}
[data-baseweb="popover"] *, [data-baseweb="menu"] *,
[role="listbox"] *, [role="option"] * {
  color:#ffffff!important; -webkit-text-fill-color:#ffffff!important;
  opacity:1!important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li, [data-baseweb="menu"] li {
  background:#14395f!important; color:#ffffff!important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] [aria-selected="true"] {
  background:#2867a0!important;
}
/* Streamlit often uses BaseWeb primitives without role=option */
[data-baseweb="popover"] [data-highlighted="true"] { background:#2867a0!important; }

[data-testid="stExpander"] {background:#112c4c!important;border:1px solid #3674a980!important;border-radius:12px;}
[data-testid="stAlert"] {background:#173d65!important;color:#eff7ff!important;border:1px solid #3e83ba70!important;}
[data-testid="stAlert"] p {color:#e8f6ff!important;}
[data-testid="stDownloadButton"] button {background:linear-gradient(100deg,#277aca,#2260ab)!important;color:white!important;border:1px solid #7dbcf890!important;}
hr {border-color:#31638e!important;}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(filename, mtime):
    d = pd.read_csv(filename, parse_dates=['datetime'])
    d['region'] = d.location.map(CITY_REGION).fillna('Other')
    return d

@st.cache_resource(show_spinner=False)
def load_selector(parameter):
    return joblib.load(MODEL_DIR / f'{parameter}_forecast_only_selector_v5.pkl')

def chart_style(fig, height=350):
    fig.update_layout(height=height, margin=dict(l=12,r=12,t=25,b=15),
       paper_bgcolor='#112c4c', plot_bgcolor='#112c4c', font=dict(color='#e4f3ff'),
       legend=dict(orientation='h', y=1.13, x=0), hovermode='x unified')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#315376', zeroline=False)
    return fig

if not LIVE.exists():
    st.error('Live forecast CSV is missing. First run: `python -u .\\scripts\\Step49B_live_31locations.py`')
    st.stop()

try:
    df = load_data(str(LIVE), LIVE.stat().st_mtime)
except Exception as ex:
    st.error(f'Could not read predictions: {ex}')
    st.stop()

required = {'location','latitude','longitude','datetime','parameter','selected_model','predicted_value','ecmwf_value'} | {f'{m}_value' for m in MODELS}
missing = sorted(required - set(df.columns))
if missing:
    st.error(f'CSV missing fields: {missing}. Run the latest Step 49B script.')
    st.stop()

# Display CSV creation age, not observation age or source-model issue time.
modified = datetime.fromtimestamp(LIVE.stat().st_mtime, ZoneInfo('Asia/Kolkata'))
st.markdown('<div class="pill">🌦️ WEATHER INTELLIGENCE · 31-LOCATION NETWORK</div>', unsafe_allow_html=True)
head_a, head_b = st.columns([3,1])
with head_a:
    st.title('Model Verification')
    st.caption('Forecast comparison • ML-estimated errors • historical ERA5 benchmark')
with head_b:
    st.markdown(f'<div style="text-align:right;padding-top:12px;font-size:13px;color:#bad9f5">CSV last refreshed<br><b>{modified:%d %b %Y, %I:%M %p} IST</b></div>', unsafe_allow_html=True)

query = st.query_params

def query_value(key, default):
    value = query.get(key, default)
    return value if isinstance(value, str) else default

regions = ['All India'] + sorted(df.region.unique())
region_default = query_value('region','All India')
if region_default not in regions: region_default='All India'

# Global filter bar
filter_cols = st.columns([1.2,1.05,1.05,1.1,1.65])
with filter_cols[0]:
    p_default = query_value('variable','temperature')
    parameter = st.selectbox('Variable', PARAMS, index=PARAMS.index(p_default) if p_default in PARAMS else 0, format_func=lambda x:LABELS[x])
with filter_cols[1]:
    horizon = st.selectbox('Lead time*', ['First available','+6h','+12h','+24h'])
with filter_cols[2]:
    period = st.selectbox('Forecast period', ['Next 12 hours','Next 24 hours','All available'])
with filter_cols[3]:
    region = st.selectbox('Region',regions,index=regions.index(region_default))
region_df = df if region=='All India' else df[df.region==region]
city_options = sorted(region_df.location.unique())
if not city_options:
    st.error('No locations available for the region.'); st.stop()
with filter_cols[4]:
    c_default = query_value('location','Karad')
    location = st.selectbox('Location',city_options,index=city_options.index(c_default) if c_default in city_options else 0,key='location_selector')

# URLs can be shared for selection; no private data included.
st.query_params.update({'variable':parameter,'region':region,'location':location})

scope = region_df[region_df.parameter==parameter].copy()
city = scope[scope.location==location].sort_values('datetime').copy()
if city.empty:
    st.warning('No data in this selection.'); st.stop()
start = city.datetime.min()
end = start + pd.Timedelta(hours={'Next 12 hours':12,'Next 24 hours':24,'All available':10000}[period])
city = city[(city.datetime>=start)&(city.datetime<end)].copy()
scope = scope[(scope.datetime>=start)&(scope.datetime<end)].copy()
if city.empty:
    st.warning('No forecast hours for this filter.'); st.stop()
hours = {'First available':0,'+6h':6,'+12h':12,'+24h':24}
requested_time = start + pd.Timedelta(hours=hours[horizon])
target_time = city.iloc[np.argmin(np.abs((city.datetime-requested_time).dt.total_seconds().to_numpy()))].datetime
selected = city[city.datetime==target_time].iloc[0]
unit=UNITS[parameter]

# Compute *estimated* expected MAE for each source model using saved trained regressors.
# Do not confuse with observational error, or the CSV's selected-model-only estimate.
@st.cache_data(show_spinner=False)
def estimate_all_models(data, parameter, model_mtime):
    saved = load_selector(parameter)
    work = data.copy()
    work['hour'] = work.datetime.dt.hour
    work['day_of_week'] = work.datetime.dt.dayofweek
    work['month'] = work.datetime.dt.month
    value_cols = [f'{m}_value' for m in MODELS]
    for m in MODELS:
        work[f'{m}_{parameter}'] = work[f'{m}_value']
    work['forecast_mean'] = work[value_cols].mean(axis=1)
    work['forecast_spread'] = work[value_cols].max(axis=1)-work[value_cols].min(axis=1)
    features = saved['features']
    valid = work[features].notna().all(axis=1)
    preds = pd.DataFrame(index=work.index, columns=MODELS, dtype=float)
    if valid.any():
        X = work.loc[valid, features]
        for m in MODELS:
            preds.loc[valid,m] = np.maximum(0, saved['models'][m].predict(X))
    return preds

model_file=MODEL_DIR / f'{parameter}_forecast_only_selector_v5.pkl'
errors=None
if model_file.exists():
    try:
        errors=estimate_all_models(scope,parameter,model_file.stat().st_mtime)
    except Exception as ex:
        st.warning(f'Estimated error comparison unavailable: {ex}')

current_expected={}
if errors is not None:
    matched=scope[(scope.location==location)&(scope.datetime==target_time)]
    if not matched.empty:
        r=errors.loc[matched.index[0]]
        current_expected={m:float(r[m]) for m in MODELS if pd.notna(r[m])}
selected_model=selected.selected_model
best_error=current_expected.get(selected_model, float(selected.get('predicted_error',np.nan)))
ecmwf_error=current_expected.get('ecmwf_ifs', np.nan)
advantage = (ecmwf_error-best_error)/ecmwf_error*100 if ecmwf_error and np.isfinite(ecmwf_error) and np.isfinite(best_error) else np.nan

city_safe=html.escape(str(location))
city_heading = city_safe
lat_value = float(selected.latitude)
lon_value = float(selected.longitude)
lat_direction = 'N' if lat_value >= 0 else 'S'
lon_direction = 'E' if lon_value >= 0 else 'W'
geo_markup = f'''<div class="geo-row">
<div class="geo-chip"><span style="font-size:25px">📍</span><div><span class="geo-label">LATITUDE</span><span class="geo-value">{abs(lat_value):.4f}° {lat_direction}</span></div></div>
<div class="geo-chip"><span style="font-size:25px">🧭</span><div><span class="geo-label">LONGITUDE</span><span class="geo-value">{abs(lon_value):.4f}° {lon_direction}</span></div></div>
</div>'''
adv_text = f'{advantage:.1f}% lower predicted error than ECMWF' if np.isfinite(advantage) and selected_model!='ecmwf_ifs' else 'ECMWF remains the selected baseline'
st.markdown(f'''<div class="hero"><div class="eyebrow">🏆 SELECTED FORECAST MODEL · {target_time:%d %b %Y %I:%M %p} IST</div>
<h2>{html.escape(NAMES.get(selected_model,selected_model))} <span style="font-size:18px;font-weight:400;opacity:.85">for {html.escape(LABELS[parameter].lower())}</span></h2>
<div class="geo-title">📍 {city_heading}, India</div>
{geo_markup}
<p>Selected forecast: <b>{float(selected.predicted_value):.2f} {unit}</b> &nbsp;·&nbsp; {html.escape(adv_text)}</p>
<div style="opacity:.75;font-size:12px;margin-top:12px">Verdict is a machine-learning estimate, NOT verified observational accuracy.</div></div>''',unsafe_allow_html=True)

# Interpret the same four predicted errors used by the live selector.
# The selection is a decision rule, not a causal attribution of meteorology.
st.markdown('<div class="section">💡 Why was this model selected?</div>', unsafe_allow_html=True)
threshold = float(selected.get('switch_threshold', 0.0))
if len(current_expected) == len(MODELS) and all(np.isfinite(v) for v in current_expected.values()):
    ranked = sorted(current_expected, key=current_expected.get)
    lowest = ranked[0]
    chosen_error = current_expected[selected_model]
    ecmwf_est = current_expected['ecmwf_ifs']
    raw_gain = ecmwf_est - current_expected[lowest]
    runner_up = next((m for m in ranked if m != selected_model), None)
    alternatives = ', '.join(
        f"{NAMES[m]} {current_expected[m]:.2f} {unit}" for m in ranked if m != selected_model
    )
    reasons = [
        f"**Expected error:** {NAMES[selected_model]} is estimated to have an absolute forecast error of **{chosen_error:.2f} {unit}** for **{location}** at **{target_time:%d %b %Y, %I:%M %p} IST**.",
        f"**Other model estimates:** {alternatives}.",
    ]
    if selected_model == 'ecmwf_ifs':
        if lowest == 'ecmwf_ifs':
            reasons.append(
                f"**Why ECMWF stays:** ECMWF has the lowest estimated error of all four models at this forecast hour."
            )
        else:
            reasons.append(
                f"**Why ECMWF stays:** {NAMES[lowest]} has a slightly lower estimated error, "
                f"but its predicted advantage is only **{raw_gain:.2f} {unit}**, "
                f"which does not exceed the required **{threshold:.2f} {unit}** switching threshold."
            )
    else:
        reasons.append(
            f"**Why the selector switched:** {NAMES[selected_model]} has the lowest estimated error and is "
            f"**{raw_gain:.2f} {unit}** lower than ECMWF's estimate of **{ecmwf_est:.2f} {unit}**. "
            f"That exceeds the **{threshold:.2f} {unit}** switching threshold."
        )
    reasons.append(
        "**What influenced the estimates:** the four current model forecasts, their average and spread, "
        "location coordinates, hour, weekday, and month. These are model inputs—not proven individual causes of the selection."
    )
    st.markdown('<div style="padding:4px 0"></div>', unsafe_allow_html=True)
    for i, reason in enumerate(reasons, start=1):
        st.markdown(f"**{i}.** {reason}")
    st.caption(
        "These are transparent decision-rule explanations based on ML-predicted errors. "
        "They do not prove which forecast will be most accurate; verification needs observed weather."
    )
else:
    st.info(
        f"Selected model: {NAMES.get(selected_model, selected_model)}. "
        "The individual model-error estimates are not available, so a detailed numerical explanation cannot be shown."
    )

# Interactive, numerical trace of the exact live inference decision.
# The saved v5 models are run again for the currently selected record.
st.markdown('<div class="section">🔎 Inside the backend · How the 4 models are compared</div>', unsafe_allow_html=True)
st.caption('Computed from the four saved v5 Random Forest regressors, for the forecast hour shown above. Smaller predicted error means a stronger candidate.')
if len(current_expected) == len(MODELS) and all(np.isfinite(v) for v in current_expected.values()):
    saved_for_trace = load_selector(parameter)
    switch_limit = float(saved_for_trace['threshold'])
    input_values = {m: float(selected[f'{m}_value']) for m in MODELS}
    sorted_models = sorted(MODELS, key=lambda m: (current_expected[m], MODELS.index(m)))
    minimum_model = sorted_models[0]
    minimum_error = current_expected[minimum_model]
    baseline_error = current_expected['ecmwf_ifs']
    predicted_gain_units = baseline_error - minimum_error
    should_switch = (minimum_model != 'ecmwf_ifs' and predicted_gain_units > switch_limit)
    decision_model = minimum_model if should_switch else 'ecmwf_ifs'

    trace_col1, trace_col2, trace_col3 = st.columns(3)
    trace_col1.metric('Lowest estimated-error model', NAMES[minimum_model])
    trace_col2.metric('Estimated gain vs ECMWF', f'{predicted_gain_units:.3f} {unit}')
    trace_col3.metric('Required switching gain', f'{switch_limit:.3f} {unit}')

    trace_rows = []
    for rank, m in enumerate(sorted_models, 1):
        trace_rows.append({
            'Rank': rank,
            'Forecast model': NAMES[m],
            f'Model forecast ({unit})': round(input_values[m], 3),
            f'ML-predicted error ({unit})': round(current_expected[m], 3),
            f'Error difference vs ECMWF ({unit})': round(current_expected[m] - baseline_error, 3),
            'Decision': '✅ SELECTED' if m == decision_model else ('Lowest error; threshold not met' if m == minimum_model else 'Not selected'),
        })
    st.dataframe(pd.DataFrame(trace_rows), hide_index=True, width='stretch')

    trace_chart=go.Figure()
    for m in sorted_models:
        trace_chart.add_trace(go.Bar(
            x=[current_expected[m]], y=[NAMES[m]], orientation='h',
            marker_color=('#7ed6ff' if m == decision_model else COLORS[m]),
            name=NAMES[m], showlegend=False,
            text=[f'{current_expected[m]:.3f} {unit}'], textposition='outside',
            hovertemplate=f'{NAMES[m]}: %{{x:.3f}} {unit}<extra></extra>'
        ))
    chart_style(trace_chart, 270)
    trace_chart.update_layout(barmode='group', xaxis_title=f'ML-predicted absolute error ({unit}) · lower is better',
                              yaxis={'categoryorder':'array', 'categoryarray':[NAMES[m] for m in reversed(sorted_models)]},
                              margin=dict(l=18,r=85,t=10,b=50))
    st.plotly_chart(trace_chart, width='stretch')

    st.markdown('**The actual decision calculation**')
    st.code(
        f'ECMWF predicted error     = {baseline_error:.4f} {unit}\n'
        f'Lowest predicted error    = {minimum_error:.4f} {unit} ({NAMES[minimum_model]})\n'
        f'Predicted gain            = {baseline_error:.4f} - {minimum_error:.4f} = {predicted_gain_units:.4f} {unit}\n'
        f'Switching threshold       = {switch_limit:.4f} {unit}\n'
        f'Switch if gain > threshold and winner != ECMWF\n'
        f'Final selected model      = {NAMES[decision_model]}',
        language='text'
    )
    if decision_model != selected_model:
        st.error(f'Decision mismatch: saved CSV selects {NAMES[selected_model]}, but the re-run selects {NAMES[decision_model]}. Check forecast or model versions before trusting this result.')
    elif should_switch:
        st.success(f'✅ Switched from ECMWF to {NAMES[decision_model]} because the predicted gain ({predicted_gain_units:.3f} {unit}) exceeds the threshold ({switch_limit:.3f} {unit}).')
    elif minimum_model == 'ecmwf_ifs':
        st.info('✅ ECMWF remains selected because it already has the lowest predicted error.')
    else:
        st.info(f'✅ ECMWF remains selected. {NAMES[minimum_model]} predicts slightly lower error, but its gain ({predicted_gain_units:.3f} {unit}) is not above the required threshold ({switch_limit:.3f} {unit}).')

    with st.expander('View the 11 feature values sent to the ML models'):
        fvals={
            f'{m}_{parameter}': input_values[m] for m in MODELS
        }
        fvals.update({
            'forecast_mean': float(np.mean(list(input_values.values()))),
            'forecast_spread': float(max(input_values.values()) - min(input_values.values())),
            'latitude': float(selected.latitude),
            'longitude': float(selected.longitude),
            'hour': int(target_time.hour),
            'day_of_week': int(target_time.dayofweek),
            'month': int(target_time.month),
        })
        ordered_features=saved_for_trace['features']
        st.dataframe(pd.DataFrame([
            {'Feature': name, 'Value passed to model': fvals.get(name, '—')}
            for name in ordered_features
        ]),hide_index=True,width='stretch')
        st.caption('These 11 values are the model inputs. The regressors learn patterns from historic forecasts versus ERA5; this view explains the decision rule, not a causal breakdown of individual tree predictions.')
else:
    st.warning('Full numerical comparison is unavailable because at least one model-specific predicted error could not be calculated.')

# Historical performance comes from a separate Aug-Sep ERA5 benchmark, NOT today's local/live observations.
hist=None
if HIST.exists():
    try:
        h=pd.read_csv(HIST)
        h=h[h.parameter==parameter]
        if not h.empty: hist=h.iloc[0]
    except Exception: pass

st.markdown('<div class="section">01 · At a glance</div>', unsafe_allow_html=True)
a,b,c,d,e=st.columns(5)
a.metric('ML-selected forecast',f'{float(selected.predicted_value):.2f} {unit}')
b.metric('Estimated selected error',f'{best_error:.2f} {unit}' if np.isfinite(best_error) else '—')
c.metric('Estimated improvement vs ECMWF',f'{advantage:+.1f}%' if np.isfinite(advantage) and selected_model!='ecmwf_ifs' else '0.0%')
d.metric('Historical MAE reduction (ERA5)',f'{float(hist.improvement_percent):+.2f}%' if hist is not None else '—')
e.metric('Forecast records in view',f'{len(city):,}')
st.caption('“Estimated” = ML-predicted absolute error at the selected forecast hour. Historical MAE reduction = Aug–Sep 2026 held-out ERA5 benchmark across locations, not city-level/live accuracy.')

# Accuracy reporting: measured historical skill must not be confused with an unobserved live forecast.
st.markdown('<div class="section">📊 Prediction performance & accuracy</div>', unsafe_allow_html=True)
if hist is not None:
    h_mae = float(hist.ml_mae)
    e_mae = float(hist.ecmwf_mae)
    skill = (e_mae - h_mae) / e_mae * 100 if e_mae > 0 else float('nan')
    k1,k2,k3 = st.columns(3)
    k1.metric('Historical ML mean absolute error', f'{h_mae:.4f} {unit}')
    k2.metric('Historical ECMWF mean absolute error', f'{e_mae:.4f} {unit}')
    k3.metric('Historical error reduction', f'{skill:+.2f}%',
              help='100 × (ECMWF MAE − ML MAE) / ECMWF MAE. This is relative skill, not percent prediction accuracy.')
    st.progress(float(max(0, min(1, skill/100))), text=f'Historical improvement relative to ECMWF: {skill:+.2f}%')
    st.caption('Measured against ERA5 reanalysis over all 31 locations in Aug–Sep 2026. '
               'These are historical aggregate scores, not an accuracy percentage for this city or this forecast hour.')
else:
    st.warning('Historical backtest summary not found. Run Step 48 to populate historical MAE metrics.')

st.info('**Live prediction accuracy: not yet verified.** The current forecast has no matched observed value. '
        'Once a valid-time-matched IMD station observation is available, we can compute observed absolute error '
        'and update station-level MAE, RMSE, and bias. No live accuracy percentage is reported yet.')
with st.expander('How is forecast accuracy calculated?'):
    st.markdown(f'''**Measured error (for one forecast):** `|forecast − observation|` in {unit}.  
**MAE:** mean of those absolute errors across matched observations. Lower means more accurate.  
**Improvement over ECMWF:** `(ECMWF MAE − ML MAE) / ECMWF MAE × 100%`.  
**Percentage accuracy:** not a universal metric for continuous weather forecasts; it requires an explicit rule such as "within ±1°C" and verified observations.  
**ML estimated error:** the regressor's prediction of likely error; it is **not** a measured score.

**Important:** The current live API may produce different forecast lead times than the 24-hour-ahead historical training samples.''')

st.markdown('<div class="section">02 · Forecast model comparison</div>',unsafe_allow_html=True)
card_cols=st.columns(4)
for i,m in enumerate(MODELS):
    value=float(selected[f'{m}_value'])
    err=current_expected.get(m,np.nan)
    with card_cols[i]:
        st.markdown(f'<div style="font-weight:800;color:{COLORS[m]};font-size:17px">● {NAMES[m]}</div>',unsafe_allow_html=True)
        st.metric('Forecast',f'{value:.2f} {unit}')
        st.caption(f'Estimated MAE: {err:.2f} {unit}' if np.isfinite(err) else 'Estimated MAE: unavailable')
        if m==selected_model: st.markdown('<span class="pill">✓ Selected at this hour</span>',unsafe_allow_html=True)

mapcol,leadercol=st.columns([1.17,1],gap='large')
with mapcol:
    st.markdown('<div class="section">03 · Station map</div>',unsafe_allow_html=True)
    plot_map=scope[scope.datetime==target_time].copy()
    if plot_map.empty:
        # Fall back to nearest hour per location for API timing anomalies.
        plot_map=scope.loc[scope.groupby('location').datetime.apply(lambda s:(s-target_time).abs().idxmin()).values]
    plot_map['model_display']=plot_map.selected_model.map(NAMES)
    plot_map['marker_color']=plot_map.selected_model.map(COLORS)
    plot_map['is_selected']=plot_map.location.eq(location)
    mapfig=go.Figure()
    for m in MODELS:
        subset=plot_map[plot_map.selected_model==m]
        if subset.empty: continue
        mapfig.add_trace(go.Scattermap(
            lat=subset.latitude,lon=subset.longitude,mode='markers',name=NAMES[m],
            marker={'size':[19 if x else 12 for x in subset.is_selected], 'color':COLORS[m], 'opacity':.88},
            customdata=subset[['location']].to_numpy(),
            hovertemplate='<b>%{customdata[0]}</b><br>Lat %{lat:.3f}, Lon %{lon:.3f}<br>Selected model: '+NAMES[m]+'<extra></extra>'
        ))
    # The active city gets a clearly identifiable cyan target pin and an on-map label.
    active_point = plot_map[plot_map.location == location]
    if not active_point.empty:
        active_point = active_point.iloc[0]
        mapfig.add_trace(go.Scattermap(
            lat=[active_point.latitude], lon=[active_point.longitude],
            mode='markers', name='Selected location',
            marker={'size':34, 'color':'#A9E8FF', 'opacity':.9},
            customdata=[[location]],
            hovertemplate='<b>📍 %{customdata[0]}</b><br>Current location selected<extra></extra>',
            showlegend=False,
        ))
        mapfig.add_trace(go.Scattermap(
            lat=[active_point.latitude],lon=[active_point.longitude],
            mode='markers+text', name='Selected: '+location,
            marker={'size':17,'color':'#083259','opacity':1},
            text=['📍 '+location], textposition='top center',
            textfont={'size':15,'color':'#FFFFFF'},
            customdata=[[location]],
            hovertemplate='<b>📍 %{customdata[0]}</b><br>Lat %{lat:.4f}, Lon %{lon:.4f}<extra></extra>',
            showlegend=False,
        ))
    mapfig.update_layout(map={'style':'carto-darkmatter','center':{'lat':21.2,'lon':79},'zoom':3.6},height=405,
       margin=dict(l=0,r=0,t=0,b=0),legend=dict(orientation='h',y=1.06),
       paper_bgcolor='#112c4c',font=dict(color='#e4f3ff'),uirevision='station-map')
    try:
        clicked=st.plotly_chart(mapfig,width='stretch',on_select='rerun',selection_mode='points',key='map_selection')
        chosen=((clicked or {}).get('selection') or {}).get('points',[])
        if chosen:
            place=chosen[0].get('customdata',[None])[0]
            if place and place in city_options and place!=location:
                st.query_params['location']=place
                st.session_state['location_selector']=place
                st.rerun()
    except TypeError:
        st.plotly_chart(mapfig,width='stretch')
    st.caption('📍 The large cyan target marker and city label show your currently selected location. Smaller colored dots represent the ML-selected model at other cities. Click any city marker to navigate.')
with leadercol:
    st.markdown('<div class="section">04 · Model leaderboard</div>',unsafe_allow_html=True)
    st.caption('Ranked by **predicted** MAE for the selected city and displayed forecast hours. This is not verified MAE.')
    if errors is not None:
        city_idxs=scope[scope.location==location].index
        e=errors.loc[city_idxs]
        leaderboard=[]
        for m in MODELS:
            vals=e[m].dropna()
            leaderboard.append({'Model':NAMES[m],'Estimated MAE':round(float(vals.mean()),3) if len(vals) else np.nan,'n hours':len(vals)})
        leaderboard=pd.DataFrame(leaderboard).sort_values('Estimated MAE').reset_index(drop=True)
        leaderboard.insert(0,'Rank',np.arange(1,len(leaderboard)+1))
        st.dataframe(leaderboard,hide_index=True,width='stretch')
    else:
        st.info('Model-specific estimates unavailable; saved selected forecasts remain visible.')
    st.info('Observed RMSE, bias, CSI, and station sample sizes require real time-aligned observations. They are intentionally not fabricated.')

st.markdown('<div class="section">05 · Hourly forecast timeline</div>',unsafe_allow_html=True)
fig=go.Figure()
for m in MODELS:
    fig.add_trace(go.Scatter(x=city.datetime,y=city[f'{m}_value'],name=NAMES[m],mode='lines',line=dict(color=COLORS[m],width=2)))
fig.add_trace(go.Scatter(x=city.datetime,y=city.predicted_value,name='ML selected',mode='lines+markers',line=dict(color='#e8f7ff',width=3,dash='dot'),marker=dict(size=5)))
chart_style(fig,390).update_yaxes(title=f'{LABELS[parameter]} ({unit})')
st.plotly_chart(fig,width='stretch')
st.caption('The dark dotted line is ML-selected forecast, not observed truth. No observed series is shown until station observations are available.')

c1,c2=st.columns(2,gap='large')
with c1:
    st.markdown('<div class="section">06 · Predicted error by forecast horizon</div>',unsafe_allow_html=True)
    if errors is not None:
        ce=errors.loc[scope[scope.location==location].index].copy()
        horizons=(scope.loc[ce.index,'datetime']-start).dt.total_seconds()/3600
        f=go.Figure()
        for m in MODELS:
            f.add_trace(go.Scatter(x=horizons,y=ce[m],name=NAMES[m],mode='lines',line=dict(color=COLORS[m])))
        chart_style(f,300).update_xaxes(title='Hours after first forecast in snapshot')
        f.update_yaxes(title=f'Estimated absolute error ({unit})')
        st.plotly_chart(f,width='stretch')
    else: st.info('Prediction-error curves unavailable.')
with c2:
    st.markdown('<div class="section">07 · Model choice by hour</div>',unsafe_allow_html=True)
    choice=city.copy()
    choice['hour_label']=choice.datetime.dt.strftime('%d %b %H:%M')
    cf=go.Figure()
    cf.add_trace(go.Bar(x=choice.datetime,y=[1]*len(choice),marker_color=choice.selected_model.map(COLORS),
      customdata=choice.selected_model.map(NAMES),hovertemplate='%{x}<br>Chosen: %{customdata}<extra></extra>',showlegend=False))
    chart_style(cf,300).update_yaxes(visible=False,range=[0,1.1]).update_xaxes(title='Forecast-valid time (IST)')
    st.plotly_chart(cf,width='stretch')
    st.caption('Color shows the model selected each hour; it does not represent verified model accuracy.')

if parameter=='precipitation':
    st.markdown('<div class="section">08 · Rainfall event perspective</div>',unsafe_allow_html=True)
    rainthreshold=st.selectbox('Rainfall event threshold (mm)',[2.5,15.5,64.5],index=0)
    counts={NAMES[m]:int((city[f'{m}_value']>=rainthreshold).sum()) for m in MODELS}
    st.write('Forecast hours predicting rainfall at or above this threshold:',counts)
    st.caption('Hit rate, false alarm rate, and CSI cannot be calculated without observed rainfall data. Forecast-hour counts are NOT verification metrics.')

with st.expander('Detailed forecast records and download'):
    cols=['datetime','location','latitude','longitude','selected_model','predicted_value','predicted_error','ecmwf_value']+[f'{m}_value' for m in MODELS if m!='ecmwf_ifs']
    st.dataframe(city[cols],hide_index=True,width='stretch')
    st.download_button('Download filtered forecasts',city.to_csv(index=False).encode('utf-8'),file_name=f'{location.replace("/","-")}_{parameter}_forecast.csv',mime='text/csv')

# Independent observational evidence, displayed separately from historical ERA5 scores.
st.markdown('<div class="section">09 · Observation stations & verification references</div>', unsafe_allow_html=True)
st.caption('Station observations are independent reference measurements. The historical ERA5 benchmark and ML-estimated errors above are different evidence sources.')

metar_file = ROOT / 'results' / 'step53_metar_actual_errors.csv'
station_names = {
    'VABB': 'Mumbai — Chhatrapati Shivaji Maharaj International Airport',
    'VIDP': 'Delhi — Indira Gandhi International Airport',
    'VOBL': 'Bengaluru — Kempegowda International Airport',
}
source_tab, imd_tab, era_tab = st.tabs(['🛬 Airport METAR (measured)', '🌧️ IMD stations (preliminary)', '📚 ERA5 (historical)'])
with source_tab:
    st.markdown('**Source:** [NOAA / Aviation Weather Center METAR API](https://aviationweather.gov/data/api/)  \n**Local comparison file:** `results/step53_metar_actual_errors.csv`')
    if metar_file.exists():
        try:
            metar = pd.read_csv(metar_file)
            required_metar = {'location','station','parameter','station_distance_km','selected_abs_error','ecmwf_ifs_abs_error'}
            if not required_metar.issubset(metar.columns):
                st.warning('METAR comparison file has unexpected columns; check Step 53 output.')
            else:
                all_stations = metar[['location','station','station_distance_km']].drop_duplicates().copy()
                all_stations['Station reference'] = all_stations['station'].map(station_names).fillna('Name not available')
                all_stations = all_stations.rename(columns={'location':'Forecast city','station':'ICAO station code','station_distance_km':'Station distance (km)'})
                st.dataframe(all_stations[['Forecast city','ICAO station code','Station reference','Station distance (km)']],hide_index=True,width='stretch')
                verified = metar[(metar['location']==location) & (metar['parameter']==parameter)].copy()
                if len(verified):
                    counts = len(verified)
                    st.markdown(f'**{location} · {LABELS[parameter]}:** {counts} matched station observations (initial exploratory comparison).')
                    err_names={'selected_abs_error':'ML selected','ecmwf_ifs_abs_error':'ECMWF','gfs_seamless_abs_error':'GFS','icon_seamless_abs_error':'ICON','gem_seamless_abs_error':'GEM'}
                    values = [{'Model':label,'Observed MAE':round(float(pd.to_numeric(verified[col], errors='coerce').mean()),3),'Matching observations':int(pd.to_numeric(verified[col],errors='coerce').count())} for col,label in err_names.items() if col in verified]
                    st.dataframe(pd.DataFrame(values),hide_index=True,width='stretch')
                    st.caption(f'Observed error units: {unit}. These preliminary scores use only a few matched hours. Station/grid differences, wind averaging conventions and original model issue times have not been fully verified. Not a reliable ranking.')
                else:
                    st.info('No matched METAR comparisons for the current city and parameter. Initial matches exist only for Mumbai, Delhi and Bengaluru, for temperature and wind speed.')
        except Exception as exc:
            st.warning(f'Could not read the saved METAR comparison: {exc}')
    else:
        st.info('The Step 53 METAR comparison CSV is not present. Run `python -u .\\scripts\\Step53_compare_metar_forecasts.py` to populate this panel.')
with imd_tab:
    st.markdown('**Source:** [India Meteorological Department (IMD)](https://mausam.imd.gov.in/)  \n**API reference:** [IMD API](https://api.imd.gov.in/public/api_reference.html)')
    st.info('An initial IMD-station comparison was carried out earlier in the project, but the verified station names, timestamps and observation-level results are not available in the current dashboard inputs. Therefore no IMD station name or measured score is invented here.')
    st.caption('Earlier limitations included imperfect forecast-observation timestamp matching and surface-pressure versus sea-level-pressure inconsistency; the latter was corrected in the forecast pipeline. Recent IMD API requests returned HTTP 401 (unauthorized).')
with era_tab:
    st.markdown('**Historical reference:** ERA5 reanalysis (model-based, not direct station measurements).')
    if hist is not None:
        st.write(f'**{LABELS[parameter]}:** ML MAE **{float(hist.ml_mae):.4f} {unit}**, ECMWF MAE **{float(hist.ecmwf_mae):.4f} {unit}**; relative error reduction **{float(hist.improvement_percent):+.2f}%**.')
    st.caption('Step 48 evaluation: Aug–Sep 2026, 31 locations. Do not label the ERA5 benchmark as IMD-verified live accuracy.')

st.divider()
st.markdown('### Data notes & methodology')
st.markdown(f'''**Forecast data:** Open-Meteo forecasts saved to CSV on **{modified:%d %b %Y, %I:%M %p} IST**. The time range shown is forecast-valid time, not verified model initialization time. *Lead time* in the filter means hours from the snapshot's first available time; it is **not** a documented forecast issue-to-valid lead time.\n\n**ML approach:** For each parameter, four Random Forest regressors estimate the absolute error of the four numerical weather model forecasts. The system switches away from ECMWF only when the predicted advantage exceeds the validation-tuned threshold.\n\n**Historical benchmark:** Step 48 held-out August–September 2026 evaluation used ERA5 reanalysis across 31 locations. This is not a live observation score, and the historical comparison is not specific to the chosen city.\n\n**Not yet available:** observed-versus-forecast charts, station-wise measured MAE / RMSE / bias, rain CSI, data latency, verified station count, and skill trends. These require properly matched IMD observations and forecast issue timestamps.\n\n**Interpretation:** Smaller model-predicted error is a *recommendation*; measured forecast error can only be known after the corresponding observation arrives.''')
