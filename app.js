/**
 * Industrial Sensor Analytics 20 - Core Application Logic
 * Pure Vanilla JavaScript - 100% Client-Side Processing, No External Transmission
 */

document.addEventListener('DOMContentLoaded', () => {
  const rawData = window.SENSOR_DATA;
  if (!rawData || rawData.length === 0) {
    console.error('No sensor data found!');
    return;
  }

  // Pre-extracted arrays for ultra-fast vector operations
  const N = rawData.length;
  const timeLabels = rawData.map(d => {
    const totalMin = Math.round(d.경과시간_분);
    const hh = String(Math.floor(totalMin / 60)).padStart(2, '0');
    const mm = String(totalMin % 60).padStart(2, '0');
    return `${hh}:${mm}`;
  });

  const minutes = rawData.map(d => d.경과시간_분);
  const tempRaw = rawData.map(d => d.온도_원시값);
  const tempTrue = rawData.map(d => d.온도_참값);
  const vibRms = rawData.map(d => d.진동_RMS);
  const pressure = rawData.map(d => d.압력);
  const events = rawData.map(d => d.이벤트);

  // Chart Registry to manage instances
  const charts = {};

  // Common Chart.js styling
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
  Chart.defaults.font.family = "'Pretendard', sans-serif";

  // Downsample helper for smooth chart rendering (every step points)
  function sample(arr, step = 2) {
    const res = [];
    for (let i = 0; i < arr.length; i += step) res.push(arr[i]);
    return res;
  }
  const sampledLabels = sample(timeLabels, 2);

  // ==========================================
  // MATHEMATICAL & STATISTICAL UTILITIES
  // ==========================================

  // Mean & StdDev
  function calcStats(arr) {
    const n = arr.length;
    const mean = arr.reduce((a, b) => a + b, 0) / n;
    const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1);
    const std = Math.sqrt(variance);
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    return { mean, std, min, max, n };
  }

  // RMSE & MAE
  function calcRMSE(pred, truth) {
    let sumSq = 0;
    for (let i = 0; i < pred.length; i++) {
      sumSq += Math.pow(pred[i] - truth[i], 2);
    }
    return Math.sqrt(sumSq / pred.length);
  }

  function calcMAE(pred, truth) {
    let sumAbs = 0;
    for (let i = 0; i < pred.length; i++) {
      sumAbs += Math.abs(pred[i] - truth[i]);
    }
    return sumAbs / pred.length;
  }

  // 1. SMA
  function computeSMA(data, windowSize) {
    const result = new Array(data.length);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i];
      if (i >= windowSize) {
        sum -= data[i - windowSize];
        result[i] = sum / windowSize;
      } else {
        result[i] = sum / (i + 1);
      }
    }
    return result;
  }

  // 2. EMA
  function computeEMA(data, alpha) {
    const result = new Array(data.length);
    result[0] = data[0];
    for (let i = 1; i < data.length; i++) {
      result[i] = alpha * data[i] + (1 - alpha) * result[i - 1];
    }
    return result;
  }

  // 3. WMA
  function computeWMA(data, windowSize) {
    const result = new Array(data.length);
    const weightSum = (windowSize * (windowSize + 1)) / 2;
    for (let i = 0; i < data.length; i++) {
      if (i < windowSize - 1) {
        // partial window
        let wSum = 0;
        let vSum = 0;
        for (let j = 0; j <= i; j++) {
          const w = j + 1;
          wSum += w;
          vSum += data[i - (i - j)] * w;
        }
        result[i] = vSum / wSum;
      } else {
        let vSum = 0;
        for (let j = 0; j < windowSize; j++) {
          const w = j + 1;
          vSum += data[i - windowSize + 1 + j] * w;
        }
        result[i] = vSum / weightSum;
      }
    }
    return result;
  }

  // 4. Savitzky-Golay Filter (Degree 2 Polynomial)
  function computeSavitzkyGolay(data, windowSize) {
    // windowSize must be odd
    if (windowSize % 2 === 0) windowSize += 1;
    const m = Math.floor(windowSize / 2);
    const coeffs = [];
    const norm = (windowSize * (windowSize * windowSize - 4)) / 3; // denominator factor
    
    // Gram polynomial weights for 2nd order smoothing
    let sumW = 0;
    for (let i = -m; i <= m; i++) {
      const w = 3 * m * m + 3 * m - 1 - 5 * i * i;
      coeffs.push(w);
      sumW += w;
    }

    const result = new Array(data.length);
    for (let i = 0; i < data.length; i++) {
      if (i < m || i >= data.length - m) {
        result[i] = data[i]; // Boundary fallback
      } else {
        let acc = 0;
        for (let j = -m; j <= m; j++) {
          acc += data[i + j] * coeffs[j + m];
        }
        result[i] = acc / sumW;
      }
    }
    return result;
  }

  // 5. 1D Kalman Filter
  function computeKalman(data, Q = 0.005, R = 0.5) {
    const result = new Array(data.length);
    let x = data[0];
    let P = 1.0;
    result[0] = x;
    for (let i = 1; i < data.length; i++) {
      // Predict
      const x_pred = x;
      const P_pred = P + Q;
      // Update
      const K = P_pred / (P_pred + R);
      x = x_pred + K * (data[i] - x_pred);
      P = (1 - K) * P_pred;
      result[i] = x;
    }
    return result;
  }

  // 6. Median Filter
  function computeMedianFilter(data, windowSize) {
    if (windowSize % 2 === 0) windowSize += 1;
    const m = Math.floor(windowSize / 2);
    const result = new Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - m);
      const end = Math.min(data.length - 1, i + m);
      const win = data.slice(start, end + 1).sort((a, b) => a - b);
      const mid = Math.floor(win.length / 2);
      result[i] = win.length % 2 === 0 ? (win[mid - 1] + win[mid]) / 2 : win[mid];
    }
    return result;
  }

  // 7. Z-Score Anomaly Detection
  function computeZScoreAnomalies(data, windowSize = 30, threshold = 2.5) {
    const mean = computeSMA(data, windowSize);
    const anomalies = [];
    const upperLimit = [];
    const lowerLimit = [];
    
    // Rolling std
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const slice = data.slice(start, i + 1);
      const m = mean[i];
      const variance = slice.reduce((acc, val) => acc + Math.pow(val - m, 2), 0) / slice.length;
      const std = Math.sqrt(variance) || 0.1;
      
      const upper = m + threshold * std;
      const lower = m - threshold * std;
      upperLimit.push(upper);
      lowerLimit.push(lower);

      if (data[i] > upper || data[i] < lower) {
        anomalies.push({ index: i, value: data[i], time: timeLabels[i] });
      }
    }
    return { anomalies, upperLimit, lowerLimit, mean };
  }

  // 8. IQR Detection
  function computeIQR(data) {
    const sorted = [...data].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const median = sorted[Math.floor(sorted.length * 0.50)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    const anomalies = [];
    for (let i = 0; i < data.length; i++) {
      if (data[i] < lowerBound || data[i] > upperBound) {
        anomalies.push({ index: i, value: data[i], time: timeLabels[i] });
      }
    }
    return { q1, median, q3, iqr, lowerBound, upperBound, anomalies };
  }

  // 9. Local Peak & Envelope Detection
  function computeEnvelopes(data, windowSize = 30) {
    const upper = new Array(data.length);
    const lower = new Array(data.length);
    const half = Math.floor(windowSize / 2);
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - half);
      const end = Math.min(data.length, i + half + 1);
      let maxV = -Infinity;
      let minV = Infinity;
      for (let j = start; j < end; j++) {
        if (data[j] > maxV) maxV = data[j];
        if (data[j] < minV) minV = data[j];
      }
      upper[i] = maxV;
      lower[i] = minV;
    }
    return { upper, lower };
  }

  // 10. Bollinger Bands
  function computeBollinger(data, period = 20, multiplier = 2) {
    const sma = computeSMA(data, period);
    const upper = new Array(data.length);
    const lower = new Array(data.length);
    const bandWidth = new Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - period + 1);
      const slice = data.slice(start, i + 1);
      const m = sma[i];
      const variance = slice.reduce((acc, v) => acc + Math.pow(v - m, 2), 0) / slice.length;
      const std = Math.sqrt(variance);
      upper[i] = m + multiplier * std;
      lower[i] = m - multiplier * std;
      bandWidth[i] = upper[i] - lower[i];
    }
    return { sma, upper, lower, bandWidth };
  }

  // 11. Additive Time Series Decomposition (Trend + Cycle + Residual)
  function computeDecomposition(data, period = 60) {
    // 1. Trend: Centered Moving Average of 60 min
    const half = Math.floor(period / 2);
    const trend = new Array(data.length);
    for (let i = 0; i < data.length; i++) {
      if (i < half || i >= data.length - half) {
        trend[i] = data[i];
      } else {
        let sum = 0;
        for (let j = i - half; j <= i + half; j++) sum += data[j];
        trend[i] = sum / (2 * half + 1);
      }
    }

    // 2. Detrended
    const detrended = data.map((d, i) => d - trend[i]);

    // 3. Cyclic / Hourly Profile (average pattern across hours)
    const hourlyAvg = new Array(60).fill(0);
    const hourlyCount = new Array(60).fill(0);
    for (let i = 0; i < detrended.length; i++) {
      const m = i % 60;
      hourlyAvg[m] += detrended[i];
      hourlyCount[m] += 1;
    }
    for (let m = 0; m < 60; m++) hourlyAvg[m] /= hourlyCount[m];

    const cyclic = data.map((_, i) => hourlyAvg[i % 60]);
    const residual = data.map((d, i) => d - trend[i] - cyclic[i]);

    return { trend, cyclic, residual };
  }

  // 12. Rate of Change (First Derivative)
  function computeROC(data) {
    const roc = new Array(data.length);
    roc[0] = 0;
    for (let i = 1; i < data.length; i++) {
      roc[i] = data[i] - data[i - 1]; // dT / dt (°C/min)
    }
    return roc;
  }

  // 13. CUSUM (Cumulative Sum Control Chart)
  function computeCUSUM(data, targetMean = 24.18, slackK = 0.5, hThreshold = 4.0) {
    const sHigh = new Array(data.length);
    const sLow = new Array(data.length);
    sHigh[0] = 0;
    sLow[0] = 0;
    const alarms = [];

    for (let i = 1; i < data.length; i++) {
      sHigh[i] = Math.max(0, sHigh[i - 1] + (data[i] - targetMean) - slackK);
      sLow[i] = Math.max(0, sLow[i - 1] - (data[i] - targetMean) - slackK);

      if (sHigh[i] > hThreshold || sLow[i] > hThreshold) {
        alarms.push({ index: i, time: timeLabels[i], high: sHigh[i], low: sLow[i] });
      }
    }
    return { sHigh, sLow, alarms };
  }

  // 14. Pearson Correlation Matrix
  function computeCorr(x, y) {
    const n = x.length;
    const mx = x.reduce((a, b) => a + b, 0) / n;
    const my = y.reduce((a, b) => a + b, 0) / n;
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - mx;
      const dy = y[i] - my;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    return num / Math.sqrt(denX * denY);
  }

  // 15. Cross-Correlation with Lag
  function computeCrossCorr(x, y, maxLag = 30) {
    const n = x.length;
    const mx = x.reduce((a, b) => a + b, 0) / n;
    const my = y.reduce((a, b) => a + b, 0) / n;
    const varX = Math.sqrt(x.reduce((a, b) => a + Math.pow(b - mx, 2), 0));
    const varY = Math.sqrt(y.reduce((a, b) => a + Math.pow(b - my, 2), 0));

    const lags = [];
    const correlations = [];

    for (let lag = -maxLag; lag <= maxLag; lag++) {
      let sum = 0;
      let count = 0;
      for (let i = 0; i < n; i++) {
        const j = i + lag;
        if (j >= 0 && j < n) {
          sum += (x[i] - mx) * (y[j] - my);
          count++;
        }
      }
      lags.push(lag);
      correlations.push(sum / (varX * varY));
    }
    return { lags, correlations };
  }

  // 16. Multi-Sensor Z-Score Normalization
  function zNormalize(arr) {
    const stats = calcStats(arr);
    return arr.map(v => (v - stats.mean) / stats.std);
  }

  // 18. Mahalanobis Distance for 3 variables (T, V, P)
  function computeMahalanobis(tArr, vArr, pArr) {
    const tStats = calcStats(tArr);
    const vStats = calcStats(vArr);
    const pStats = calcStats(pArr);

    // Centered matrix
    const n = tArr.length;
    const tC = tArr.map(v => v - tStats.mean);
    const vC = vArr.map(v => v - vStats.mean);
    const pC = pArr.map(v => v - pStats.mean);

    // 3x3 Covariance matrix
    let cTT = 0, cTV = 0, cTP = 0, cVV = 0, cVP = 0, cPP = 0;
    for (let i = 0; i < n; i++) {
      cTT += tC[i] * tC[i];
      cTV += tC[i] * vC[i];
      cTP += tC[i] * pC[i];
      cVV += vC[i] * vC[i];
      cVP += vC[i] * pC[i];
      cPP += pC[i] * pC[i];
    }
    cTT /= (n - 1); cTV /= (n - 1); cTP /= (n - 1);
    cVV /= (n - 1); cVP /= (n - 1); cPP /= (n - 1);

    // Invert 3x3 symmetric matrix
    // A = [ [cTT, cTV, cTP], [cTV, cVV, cVP], [cTP, cVP, cPP] ]
    const det = cTT * (cVV * cPP - cVP * cVP) - cTV * (cTV * cPP - cVP * cTP) + cTP * (cTV * cVP - cVV * cTP);
    const inv = [
      [(cVV * cPP - cVP * cVP) / det, (cTP * cVP - cTV * cPP) / det, (cTV * cVP - cTP * cVV) / det],
      [(cTP * cVP - cTV * cPP) / det, (cTT * cPP - cTP * cTP) / det, (cTP * cTV - cTT * cVP) / det],
      [(cTV * cVP - cTP * cVV) / det, (cTP * cTV - cTT * cVP) / det, (cTT * cVV - cTV * cTV) / det]
    ];

    const distances = new Array(n);
    for (let i = 0; i < n; i++) {
      const dt = tC[i];
      const dv = vC[i];
      const dp = pC[i];

      const u0 = dt * inv[0][0] + dv * inv[1][0] + dp * inv[2][0];
      const u1 = dt * inv[0][1] + dv * inv[1][1] + dp * inv[2][1];
      const u2 = dt * inv[0][2] + dv * inv[1][2] + dp * inv[2][2];

      const d2 = u0 * dt + u1 * dv + u2 * dp;
      distances[i] = Math.sqrt(Math.max(0, d2));
    }
    return distances;
  }

  // 20. FFT / Discrete Fourier Transform (DFT Amplitude Spectrum)
  function computeDFT(data, numPoints = 256) {
    // Zero-mean signal
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const signal = data.slice(0, numPoints).map(v => v - mean);
    const freqs = [];
    const amplitudes = [];
    const N_dft = signal.length;

    for (let k = 0; k < N_dft / 2; k++) {
      let real = 0;
      let imag = 0;
      for (let n = 0; n < N_dft; n++) {
        const angle = (2 * Math.PI * k * n) / N_dft;
        real += signal[n] * Math.cos(angle);
        imag -= signal[n] * Math.sin(angle);
      }
      const amp = (2 * Math.sqrt(real * real + imag * imag)) / N_dft;
      // Cycles per day (or per hour)
      // Sampling rate is 1 sample / min = 1440 samples / day
      const cyclesPerDay = (k * 1440) / N_dft;
      freqs.push(cyclesPerDay.toFixed(1) + ' cyc/day');
      amplitudes.push(amp);
    }
    return { freqs, amplitudes };
  }

  // ==========================================
  // RENDER KPI HERO SUMMARY
  // ==========================================
  function renderKPISummary() {
    const rawStats = calcStats(tempRaw);
    const trueStats = calcStats(tempTrue);
    const vibStats = calcStats(vibRms);
    const pressStats = calcStats(pressure);

    const initialRMSE = calcRMSE(tempRaw, tempTrue);
    const sgFiltered = computeSavitzkyGolay(tempRaw, 21);
    const optimalRMSE = calcRMSE(sgFiltered, tempTrue);

    // Count non-normal events
    const abnormalCount = events.filter(e => e !== '정상').length;

    document.getElementById('kpi-total-points').textContent = N.toLocaleString();
    document.getElementById('kpi-abnormal-count').textContent = `${abnormalCount} min (${((abnormalCount / N) * 100).toFixed(1)}%)`;
    document.getElementById('kpi-rmse-baseline').textContent = `${initialRMSE.toFixed(3)} °C`;
    document.getElementById('kpi-rmse-optimal').textContent = `${optimalRMSE.toFixed(3)} °C (-${(((initialRMSE - optimalRMSE) / initialRMSE) * 100).toFixed(1)}%)`;
    document.getElementById('kpi-max-temp').textContent = `${rawStats.max.toFixed(2)} °C`;
    document.getElementById('kpi-avg-vib').textContent = `${vibStats.mean.toFixed(2)} mm/s`;
  }

  // ==========================================
  // INITIALIZE ALL 20 TECHNIQUES
  // ==========================================

  // --- 1. SMA ---
  function initSMA() {
    const select = document.getElementById('sma-window');
    const update = () => {
      const w = parseInt(select.value, 10);
      const smaData = computeSMA(tempRaw, w);
      const rmse = calcRMSE(smaData, tempTrue);
      document.getElementById('sma-rmse').textContent = `${rmse.toFixed(3)} °C`;

      const chartData = {
        labels: sampledLabels,
        datasets: [
          { label: '원시 온도', data: sample(tempRaw), borderColor: 'rgba(148, 163, 184, 0.3)', borderWidth: 1, pointRadius: 0 },
          { label: '참값(Ground Truth)', data: sample(tempTrue), borderColor: '#10b981', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0 },
          { label: `SMA (${w}분)`, data: sample(smaData), borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0 }
        ]
      };

      if (charts.sma) charts.sma.destroy();
      charts.sma = new Chart(document.getElementById('chart-sma'), {
        type: 'line',
        data: chartData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false }
        }
      });
    };
    select.addEventListener('change', update);
    update();
  }

  // --- 2. EMA ---
  function initEMA() {
    const slider = document.getElementById('ema-alpha');
    const label = document.getElementById('ema-alpha-val');
    const update = () => {
      const alpha = parseFloat(slider.value);
      label.textContent = alpha.toFixed(2);
      const emaData = computeEMA(tempRaw, alpha);
      const rmse = calcRMSE(emaData, tempTrue);
      document.getElementById('ema-rmse').textContent = `${rmse.toFixed(3)} °C`;

      const chartData = {
        labels: sampledLabels,
        datasets: [
          { label: '원시 온도', data: sample(tempRaw), borderColor: 'rgba(148, 163, 184, 0.3)', borderWidth: 1, pointRadius: 0 },
          { label: '참값(Ground Truth)', data: sample(tempTrue), borderColor: '#10b981', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0 },
          { label: `EMA (α=${alpha})`, data: sample(emaData), borderColor: '#f97316', borderWidth: 2, pointRadius: 0 }
        ]
      };

      if (charts.ema) charts.ema.destroy();
      charts.ema = new Chart(document.getElementById('chart-ema'), {
        type: 'line',
        data: chartData,
        options: { responsive: true, maintainAspectRatio: false }
      });
    };
    slider.addEventListener('input', update);
    update();
  }

  // --- 3. WMA ---
  function initWMA() {
    const select = document.getElementById('wma-window');
    const update = () => {
      const w = parseInt(select.value, 10);
      const wmaData = computeWMA(tempRaw, w);
      const smaData = computeSMA(tempRaw, w);
      const rmseWMA = calcRMSE(wmaData, tempTrue);
      const rmseSMA = calcRMSE(smaData, tempTrue);

      document.getElementById('wma-rmse').textContent = `${rmseWMA.toFixed(3)} °C`;
      document.getElementById('wma-vs-sma').textContent = `${((rmseSMA - rmseWMA) > 0 ? '-' : '+')}${Math.abs(rmseSMA - rmseWMA).toFixed(3)} °C`;

      if (charts.wma) charts.wma.destroy();
      charts.wma = new Chart(document.getElementById('chart-wma'), {
        type: 'line',
        data: {
          labels: sampledLabels,
          datasets: [
            { label: '참값', data: sample(tempTrue), borderColor: '#10b981', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0 },
            { label: `WMA (${w}분 가중)`, data: sample(wmaData), borderColor: '#8b5cf6', borderWidth: 2, pointRadius: 0 },
            { label: `SMA (${w}분 단순)`, data: sample(smaData), borderColor: 'rgba(6, 182, 212, 0.6)', borderWidth: 1.5, pointRadius: 0 }
          ]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    };
    select.addEventListener('change', update);
    update();
  }

  // --- 4. Savitzky-Golay ---
  function initSavitzkyGolay() {
    const select = document.getElementById('sg-window');
    const update = () => {
      const w = parseInt(select.value, 10);
      const sgData = computeSavitzkyGolay(tempRaw, w);
      const smaData = computeSMA(tempRaw, w);
      const rmse = calcRMSE(sgData, tempTrue);
      document.getElementById('sg-rmse').textContent = `${rmse.toFixed(3)} °C`;

      // Peak preservation check around minute 784~796 (acute overheat)
      const maxTrue = Math.max(...tempTrue.slice(780, 800));
      const maxSG = Math.max(...sgData.slice(780, 800));
      const maxSMA = Math.max(...smaData.slice(780, 800));
      document.getElementById('sg-peak-error').textContent = `SG: ${(maxSG - maxTrue).toFixed(2)}°C vs SMA: ${(maxSMA - maxTrue).toFixed(2)}°C`;

      if (charts.sg) charts.sg.destroy();
      charts.sg = new Chart(document.getElementById('chart-sg'), {
        type: 'line',
        data: {
          labels: sampledLabels,
          datasets: [
            { label: '원시값', data: sample(tempRaw), borderColor: 'rgba(148, 163, 184, 0.25)', borderWidth: 1, pointRadius: 0 },
            { label: '참값(Truth)', data: sample(tempTrue), borderColor: '#10b981', borderWidth: 2, borderDash: [4, 4], pointRadius: 0 },
            { label: `Savitzky-Golay (윈도우 ${w})`, data: sample(sgData), borderColor: '#38bdf8', borderWidth: 2.5, pointRadius: 0 }
          ]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    };
    select.addEventListener('change', update);
    update();
  }

  // --- 5. Kalman Filter ---
  function initKalman() {
    const sliderR = document.getElementById('kalman-r');
    const labelR = document.getElementById('kalman-r-val');
    const update = () => {
      const rVal = parseFloat(sliderR.value);
      labelR.textContent = rVal.toFixed(2);
      const kalmanData = computeKalman(tempRaw, 0.01, rVal);
      const rmse = calcRMSE(kalmanData, tempTrue);
      document.getElementById('kalman-rmse').textContent = `${rmse.toFixed(3)} °C`;

      if (charts.kalman) charts.kalman.destroy();
      charts.kalman = new Chart(document.getElementById('chart-kalman'), {
        type: 'line',
        data: {
          labels: sampledLabels,
          datasets: [
            { label: '원시값', data: sample(tempRaw), borderColor: 'rgba(148, 163, 184, 0.3)', borderWidth: 1, pointRadius: 0 },
            { label: '참값', data: sample(tempTrue), borderColor: '#10b981', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0 },
            { label: `Kalman 추정치 (R=${rVal})`, data: sample(kalmanData), borderColor: '#ec4899', borderWidth: 2, pointRadius: 0 }
          ]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    };
    sliderR.addEventListener('input', update);
    update();
  }

  // --- 6. Median & Hampel Filter ---
  function initMedian() {
    const select = document.getElementById('median-window');
    const update = () => {
      const w = parseInt(select.value, 10);
      const medData = computeMedianFilter(tempRaw, w);
      const rmse = calcRMSE(medData, tempTrue);
      document.getElementById('med-rmse').textContent = `${rmse.toFixed(3)} °C`;

      // Check spike rejection at minute 188
      const spikeDiff = Math.abs(medData[188] - tempRaw[188]);
      document.getElementById('med-spike-reject').textContent = `${spikeDiff.toFixed(2)} °C 제거`;

      if (charts.median) charts.median.destroy();
      charts.median = new Chart(document.getElementById('chart-median'), {
        type: 'line',
        data: {
          labels: sampledLabels,
          datasets: [
            { label: '원시 신호 (스파이크 포함)', data: sample(tempRaw), borderColor: 'rgba(239, 68, 68, 0.4)', borderWidth: 1, pointRadius: 0 },
            { label: '참값', data: sample(tempTrue), borderColor: '#10b981', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0 },
            { label: `중앙값 필터 (${w}분 창)`, data: sample(medData), borderColor: '#eab308', borderWidth: 2, pointRadius: 0 }
          ]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    };
    select.addEventListener('change', update);
    update();
  }

  // --- 7. Z-Score Anomaly Detection ---
  function initZScore() {
    const slider = document.getElementById('zscore-thresh');
    const label = document.getElementById('zscore-thresh-val');
    const update = () => {
      const th = parseFloat(slider.value);
      label.textContent = `${th.toFixed(1)} σ`;
      const { anomalies, upperLimit, lowerLimit, mean } = computeZScoreAnomalies(tempRaw, 30, th);
      document.getElementById('zscore-count').textContent = `${anomalies.length} 건`;

      const anomalyScatter = sampledLabels.map((lbl, idx) => {
        const origIdx = idx * 2;
        const hit = anomalies.find(a => a.index === origIdx);
        return hit ? hit.value : null;
      });

      if (charts.zscore) charts.zscore.destroy();
      charts.zscore = new Chart(document.getElementById('chart-zscore'), {
        type: 'line',
        data: {
          labels: sampledLabels,
          datasets: [
            { label: '원시 온도', data: sample(tempRaw), borderColor: 'rgba(148, 163, 184, 0.4)', borderWidth: 1, pointRadius: 0 },
            { label: '상한 밴드 (+kσ)', data: sample(upperLimit), borderColor: 'rgba(239, 68, 68, 0.5)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0 },
            { label: '하한 밴드 (-kσ)', data: sample(lowerLimit), borderColor: 'rgba(239, 68, 68, 0.5)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0 },
            { label: '이상치 탐지점', data: anomalyScatter, borderColor: '#ef4444', backgroundColor: '#ef4444', pointRadius: 5, showLine: false }
          ]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    };
    slider.addEventListener('input', update);
    update();
  }

  // --- 8. IQR Outlier Detection ---
  function initIQR() {
    const { q1, median, q3, iqr, lowerBound, upperBound, anomalies } = computeIQR(tempRaw);
    if (document.getElementById('iqr-q1')) document.getElementById('iqr-q1').textContent = `${q1.toFixed(2)} °C`;
    if (document.getElementById('iqr-median')) document.getElementById('iqr-median').textContent = `${q1.toFixed(1)} / ${median.toFixed(1)} / ${q3.toFixed(1)} °C`;
    if (document.getElementById('iqr-q3')) document.getElementById('iqr-q3').textContent = `${q3.toFixed(2)} °C`;
    if (document.getElementById('iqr-bounds')) document.getElementById('iqr-bounds').textContent = `[${lowerBound.toFixed(2)} ~ ${upperBound.toFixed(2)}] °C`;
    if (document.getElementById('iqr-count')) document.getElementById('iqr-count').textContent = `${anomalies.length} 건`;

    // Histogram of raw temperature
    const binCount = 25;
    const minV = Math.floor(Math.min(...tempRaw));
    const maxV = Math.ceil(Math.max(...tempRaw));
    const binSize = (maxV - minV) / binCount;
    const bins = new Array(binCount).fill(0);
    const binLabels = [];
    for (let i = 0; i < binCount; i++) {
      binLabels.push((minV + i * binSize).toFixed(1));
    }
    tempRaw.forEach(v => {
      let b = Math.floor((v - minV) / binSize);
      if (b >= binCount) b = binCount - 1;
      bins[b]++;
    });

    const bgColors = binLabels.map(l => {
      const val = parseFloat(l);
      return (val < lowerBound || val > upperBound) ? 'rgba(239, 68, 68, 0.7)' : 'rgba(59, 130, 246, 0.6)';
    });

    if (charts.iqr) charts.iqr.destroy();
    charts.iqr = new Chart(document.getElementById('chart-iqr'), {
      type: 'bar',
      data: {
        labels: binLabels,
        datasets: [{
          label: '온도 빈도수 분포 (적색: IQR 이상치 구간)',
          data: bins,
          backgroundColor: bgColors,
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: true } },
        scales: { x: { title: { display: true, text: '온도 구간 (°C)' } } }
      }
    });
  }

  // --- 9. Peak & Envelopes ---
  function initPeakEnvelopes() {
    const select = document.getElementById('env-window');
    const update = () => {
      const w = parseInt(select.value, 10);
      const { upper, lower } = computeEnvelopes(tempRaw, w);
      const avgSpan = upper.reduce((acc, u, i) => acc + (u - lower[i]), 0) / upper.length;
      document.getElementById('env-avg-span').textContent = `${avgSpan.toFixed(2)} °C`;

      if (charts.env) charts.env.destroy();
      charts.env = new Chart(document.getElementById('chart-env'), {
        type: 'line',
        data: {
          labels: sampledLabels,
          datasets: [
            { label: '상한 피크 포락선 (Max)', data: sample(upper), borderColor: '#f43f5e', borderWidth: 1.5, pointRadius: 0 },
            { label: '원시 센서 신호', data: sample(tempRaw), borderColor: 'rgba(148, 163, 184, 0.35)', borderWidth: 1, pointRadius: 0 },
            { label: '하한 밸리 포락선 (Min)', data: sample(lower), borderColor: '#06b6d4', borderWidth: 1.5, pointRadius: 0 }
          ]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    };
    select.addEventListener('change', update);
    update();
  }

  // --- 10. Bollinger Bands ---
  function initBollinger() {
    const { sma, upper, lower, bandWidth } = computeBollinger(tempRaw, 20, 2);
    const maxBand = Math.max(...bandWidth);
    document.getElementById('bb-max-width').textContent = `${maxBand.toFixed(2)} °C`;

    if (charts.bb) charts.bb.destroy();
    charts.bb = new Chart(document.getElementById('chart-bb'), {
      type: 'line',
      data: {
        labels: sampledLabels,
        datasets: [
          { label: '상한 밴드 (+2σ)', data: sample(upper), borderColor: 'rgba(245, 158, 11, 0.8)', borderWidth: 1.2, pointRadius: 0 },
          { label: '중심선 (20분 SMA)', data: sample(sma), borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0 },
          { label: '하한 밴드 (-2σ)', data: sample(lower), borderColor: 'rgba(245, 158, 11, 0.8)', borderWidth: 1.2, pointRadius: 0 },
          { label: '원시 온도', data: sample(tempRaw), borderColor: 'rgba(148, 163, 184, 0.3)', borderWidth: 1, pointRadius: 0 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  // --- 11. Time Series Decomposition ---
  function initDecomp() {
    const { trend, cyclic, residual } = computeDecomposition(tempRaw, 60);

    if (charts.decomp) charts.decomp.destroy();
    charts.decomp = new Chart(document.getElementById('chart-decomp'), {
      type: 'line',
      data: {
        labels: sampledLabels,
        datasets: [
          { label: '기저 추세 성분 (Trend)', data: sample(trend), borderColor: '#f97316', borderWidth: 2, pointRadius: 0 },
          { label: '시간 주기 성분 (Cyclic)', data: sample(cyclic), borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0 },
          { label: '불규칙 잔차 (Residual)', data: sample(residual), borderColor: 'rgba(148, 163, 184, 0.5)', borderWidth: 1, pointRadius: 0 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  // --- 12. Rate of Change (First Derivative) ---
  function initROC() {
    const roc = computeROC(tempRaw);
    const maxROC = Math.max(...roc);
    const minROC = Math.min(...roc);
    document.getElementById('roc-max').textContent = `+${maxROC.toFixed(2)} °C/min`;
    document.getElementById('roc-min').textContent = `${minROC.toFixed(2)} °C/min`;

    if (charts.roc) charts.roc.destroy();
    charts.roc = new Chart(document.getElementById('chart-roc'), {
      type: 'line',
      data: {
        labels: sampledLabels,
        datasets: [
          { label: '온도 분당 변화율 (dT/dt)', data: sample(roc), borderColor: '#a855f7', borderWidth: 1.2, pointRadius: 0 },
          { label: '경고 임계치 (+1.5°C/분)', data: new Array(sampledLabels.length).fill(1.5), borderColor: 'rgba(239, 68, 68, 0.6)', borderDash: [4, 4], pointRadius: 0 },
          { label: '경고 임계치 (-1.5°C/분)', data: new Array(sampledLabels.length).fill(-1.5), borderColor: 'rgba(6, 182, 212, 0.6)', borderDash: [4, 4], pointRadius: 0 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  // --- 13. CUSUM Control Chart ---
  function initCUSUM() {
    const { sHigh, sLow, alarms } = computeCUSUM(tempRaw, 24.18, 0.4, 4.0);
    document.getElementById('cusum-alarms').textContent = `${alarms.length} 건 지속`;

    if (charts.cusum) charts.cusum.destroy();
    charts.cusum = new Chart(document.getElementById('chart-cusum'), {
      type: 'line',
      data: {
        labels: sampledLabels,
        datasets: [
          { label: 'CUSUM+ (상승 누적)', data: sample(sHigh), borderColor: '#f43f5e', borderWidth: 2, pointRadius: 0 },
          { label: 'CUSUM- (하강 누적)', data: sample(sLow), borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0 },
          { label: '관리 한계선 (h=4.0)', data: new Array(sampledLabels.length).fill(4.0), borderColor: '#eab308', borderDash: [4, 4], pointRadius: 0 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  // --- 14. Correlation Matrix ---
  function initCorrelation() {
    const cTT = computeCorr(tempRaw, tempTrue);
    const cTV = computeCorr(tempTrue, vibRms);
    const cTP = computeCorr(tempTrue, pressure);
    const cVP = computeCorr(vibRms, pressure);
    const cRV = computeCorr(tempRaw, vibRms);
    const cRP = computeCorr(tempRaw, pressure);

    if (document.getElementById('corr-tt')) document.getElementById('corr-tt').textContent = cTT.toFixed(3);
    if (document.getElementById('corr-tv')) document.getElementById('corr-tv').textContent = cTV.toFixed(3);
    if (document.getElementById('corr-tp')) document.getElementById('corr-tp').textContent = cTP.toFixed(3);
    if (document.getElementById('corr-vp')) document.getElementById('corr-vp').textContent = cVP.toFixed(3);
    if (document.getElementById('corr-rv')) document.getElementById('corr-rv').textContent = cRV.toFixed(3);
    if (document.getElementById('corr-rp')) document.getElementById('corr-rp').textContent = cRP.toFixed(3);

    // Scatter plot: Temperature vs Vibration
    const scatterData = [];
    for (let i = 0; i < N; i += 3) {
      scatterData.push({ x: tempTrue[i], y: vibRms[i] });
    }

    if (charts.corr) charts.corr.destroy();
    charts.corr = new Chart(document.getElementById('chart-corr'), {
      type: 'scatter',
      data: {
        datasets: [{
          label: '온도(참값) vs 진동 RMS 산점도 (r = 0.786)',
          data: scatterData,
          backgroundColor: 'rgba(6, 182, 212, 0.6)',
          pointRadius: 2.5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: '온도 (°C)' } },
          y: { title: { display: true, text: '진동 RMS (mm/s)' } }
        }
      }
    });
  }

  // --- 15. Cross-Correlation & Lag ---
  function initCrossCorrelation() {
    const { lags, correlations } = computeCrossCorr(vibRms, tempTrue, 30);
    let maxR = -Infinity;
    let optLag = 0;
    for (let i = 0; i < lags.length; i++) {
      if (correlations[i] > maxR) {
        maxR = correlations[i];
        optLag = lags[i];
      }
    }
    document.getElementById('lag-optimal').textContent = `${optLag} 분 (상관계수 ${maxR.toFixed(3)})`;

    if (charts.crossCorr) charts.crossCorr.destroy();
    charts.crossCorr = new Chart(document.getElementById('chart-crosscorr'), {
      type: 'bar',
      data: {
        labels: lags.map(l => `${l}분`),
        datasets: [{
          label: '진동과 온도의 시차(Lag)별 교차 상관계수',
          data: correlations,
          backgroundColor: lags.map(l => l === optLag ? '#f59e0b' : 'rgba(59, 130, 246, 0.6)'),
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: '진동 선행/지연 시차 (분)' } },
          y: { min: 0.5, max: 0.85, title: { display: true, text: '교차상관계수 R' } }
        }
      }
    });
  }

  // --- 16. Multi-Sensor Normalized Overlay ---
  function initMultiSensor() {
    const zTemp = zNormalize(tempTrue);
    const zVib = zNormalize(vibRms);
    const zPress = zNormalize(pressure);

    if (charts.multi) charts.multi.destroy();
    charts.multi = new Chart(document.getElementById('chart-multisensor'), {
      type: 'line',
      data: {
        labels: sampledLabels,
        datasets: [
          { label: '표준화 온도 (z-Score)', data: sample(zTemp), borderColor: '#f43f5e', borderWidth: 1.8, pointRadius: 0 },
          { label: '표준화 진동 (z-Score)', data: sample(zVib), borderColor: '#06b6d4', borderWidth: 1.8, pointRadius: 0 },
          { label: '표준화 압력 (z-Score)', data: sample(zPress), borderColor: '#eab308', borderWidth: 1.5, pointRadius: 0 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  // --- 17. Phase Space State Trajectory ---
  function initPhaseSpace() {
    // 2D Phase Trajectory: Temperature vs Vibration, colored by Event
    const colorMap = {
      '정상': 'rgba(16, 185, 129, 0.4)',
      '스파이크 이상치': '#ef4444',
      '단기 과열': '#fb923c',
      '부하 증가': '#8b5cf6',
      '급격한 과열': '#dc2626',
      '장시간 부하': '#a855f7',
      '일시 냉각': '#06b6d4'
    };

    const phaseData = [];
    for (let i = 0; i < N; i += 2) {
      phaseData.push({
        x: tempRaw[i],
        y: vibRms[i],
        ev: events[i]
      });
    }

    if (charts.phase) charts.phase.destroy();
    charts.phase = new Chart(document.getElementById('chart-phase'), {
      type: 'scatter',
      data: {
        datasets: [{
          label: '상태 공간 점 (온도 vs 진동)',
          data: phaseData,
          backgroundColor: phaseData.map(p => colorMap[p.ev] || 'rgba(148, 163, 184, 0.4)'),
          pointRadius: phaseData.map(p => p.ev === '정상' ? 2 : 5)
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: '온도 (°C)' } },
          y: { title: { display: true, text: '진동 RMS (mm/s)' } }
        }
      }
    });
  }

  // --- 18. Mahalanobis Health Index ---
  function initMahalanobis() {
    const distances = computeMahalanobis(tempRaw, vibRms, pressure);
    const threshold = 3.5;
    const abnormalCount = distances.filter(d => d > threshold).length;
    document.getElementById('mah-count').textContent = `${abnormalCount} 건`;

    if (charts.mah) charts.mah.destroy();
    charts.mah = new Chart(document.getElementById('chart-mahalanobis'), {
      type: 'line',
      data: {
        labels: sampledLabels,
        datasets: [
          { label: '마할라노비스 건전성 거리 (D_M)', data: sample(distances), borderColor: '#f97316', borderWidth: 1.5, pointRadius: 0 },
          { label: '임계치 (D=3.5)', data: new Array(sampledLabels.length).fill(threshold), borderColor: '#ef4444', borderDash: [4, 4], pointRadius: 0 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  // --- 19. Filter Benchmark ---
  function initBenchmark() {
    const filters = [
      { name: '원시값 (No Filter)', data: tempRaw },
      { name: 'SMA (5분)', data: computeSMA(tempRaw, 5) },
      { name: 'SMA (15분)', data: computeSMA(tempRaw, 15) },
      { name: 'EMA (α=0.2)', data: computeEMA(tempRaw, 0.2) },
      { name: 'WMA (15분)', data: computeWMA(tempRaw, 15) },
      { name: 'Savitzky-Golay (21분)', data: computeSavitzkyGolay(tempRaw, 21) },
      { name: '1D Kalman (R=0.5)', data: computeKalman(tempRaw, 0.01, 0.5) },
      { name: 'Median (5분)', data: computeMedianFilter(tempRaw, 5) }
    ];

    const results = filters.map(f => {
      const rmse = calcRMSE(f.data, tempTrue);
      const mae = calcMAE(f.data, tempTrue);
      let maxErr = 0;
      for (let i = 0; i < N; i++) {
        const err = Math.abs(f.data[i] - tempTrue[i]);
        if (err > maxErr) maxErr = err;
      }
      return { name: f.name, rmse, mae, maxErr };
    });

    // Sort by RMSE ascending
    results.sort((a, b) => a.rmse - b.rmse);

    const tbody = document.getElementById('benchmark-tbody');
    tbody.innerHTML = '';
    results.forEach((r, idx) => {
      const tr = document.createElement('tr');
      const rankClass = idx === 0 ? 'rank-1' : idx === 1 ? 'rank-2' : idx === 2 ? 'rank-3' : '';
      tr.innerHTML = `
        <td><span class="rank-pill ${rankClass}">#${idx + 1}</span></td>
        <td><strong>${r.name}</strong></td>
        <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${r.rmse.toFixed(3)} °C</td>
        <td style="font-family: var(--font-mono);">${r.mae.toFixed(3)} °C</td>
        <td style="font-family: var(--font-mono); color: #f87171;">${r.maxErr.toFixed(2)} °C</td>
      `;
      tbody.appendChild(tr);
    });

    if (charts.bench) charts.bench.destroy();
    charts.bench = new Chart(document.getElementById('chart-benchmark'), {
      type: 'bar',
      data: {
        labels: results.map(r => r.name),
        datasets: [{
          label: '참값 대비 RMSE 오차 (°C, 낮을수록 우수)',
          data: results.map(r => r.rmse),
          backgroundColor: results.map((_, i) => i === 0 ? '#10b981' : i === 1 ? '#06b6d4' : 'rgba(59, 130, 246, 0.6)'),
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } }
      }
    });
  }

  // --- 20. FFT / Power Spectral Density ---
  function initFFT() {
    const { freqs, amplitudes } = computeDFT(tempRaw, 256);

    if (charts.fft) charts.fft.destroy();
    charts.fft = new Chart(document.getElementById('chart-fft'), {
      type: 'line',
      data: {
        labels: freqs,
        datasets: [{
          label: '주파수 스펙트럼 진폭 (DFT Amplitude)',
          data: amplitudes,
          borderColor: '#a855f7',
          backgroundColor: 'rgba(168, 85, 247, 0.2)',
          fill: true,
          borderWidth: 1.5,
          pointRadius: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: '주파수 (일주기 당 반복 횟수, Cycles/Day)' } },
          y: { title: { display: true, text: '진폭 스펙트럼 크기' } }
        }
      }
    });
  }

  // ==========================================
  // RAW DATA TABLE VIEWER & SEARCH & CSV EXPORT
  // ==========================================
  function initDataTable() {
    const tbody = document.getElementById('raw-table-tbody');
    const searchInput = document.getElementById('table-search');
    const eventFilter = document.getElementById('table-event-filter');
    const pageInfo = document.getElementById('table-page-info');
    const btnPrev = document.getElementById('table-prev');
    const btnNext = document.getElementById('table-next');

    let currentPage = 1;
    const pageSize = 20;
    let filteredData = rawData;

    function applyFilter() {
      const q = searchInput.value.trim().toLowerCase();
      const ev = eventFilter.value;

      filteredData = rawData.filter((d, idx) => {
        const timeStr = timeLabels[idx];
        const matchText = timeStr.includes(q) || String(d.경과시간_분).includes(q);
        const matchEvent = ev === 'ALL' || d.이벤트 === ev;
        return matchText && matchEvent;
      });

      currentPage = 1;
      renderTable();
    }

    function renderTable() {
      const totalPages = Math.ceil(filteredData.length / pageSize) || 1;
      pageInfo.textContent = `${currentPage} / ${totalPages} (총 ${filteredData.length} 건)`;
      btnPrev.disabled = currentPage <= 1;
      btnNext.disabled = currentPage >= totalPages;

      tbody.innerHTML = '';
      const start = (currentPage - 1) * pageSize;
      const slice = filteredData.slice(start, start + pageSize);

      slice.forEach(d => {
        const tr = document.createElement('tr');
        const totalMin = Math.round(d.경과시간_분);
        const hh = String(Math.floor(totalMin / 60)).padStart(2, '0');
        const mm = String(totalMin % 60).padStart(2, '0');
        const timeStr = `${hh}:${mm}`;

        let eventClass = 'event-normal';
        if (d.이벤트.includes('스파이크')) eventClass = 'event-spike';
        else if (d.이벤트.includes('과열')) eventClass = 'event-overheat';
        else if (d.이벤트.includes('부하')) eventClass = 'event-load';
        else if (d.이벤트.includes('냉각')) eventClass = 'event-cool';

        tr.innerHTML = `
          <td style="font-family: var(--font-mono); font-weight: 600;">${timeStr}</td>
          <td>${d.경과시간_분}</td>
          <td style="font-family: var(--font-mono); color: ${Math.abs(d.온도_원시값 - d.온도_참값) > 2 ? '#f87171' : 'inherit'};">${d.온도_원시값.toFixed(3)}</td>
          <td style="font-family: var(--font-mono); color: #34d399;">${d.온도_참값.toFixed(3)}</td>
          <td style="font-family: var(--font-mono);">${d.진동_RMS.toFixed(3)}</td>
          <td style="font-family: var(--font-mono);">${d.압력.toFixed(3)}</td>
          <td><span class="event-tag ${eventClass}">${d.이벤트}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    searchInput.addEventListener('input', applyFilter);
    eventFilter.addEventListener('change', applyFilter);

    btnPrev.addEventListener('click', () => {
      if (currentPage > 1) { currentPage--; renderTable(); }
    });
    btnNext.addEventListener('click', () => {
      const totalPages = Math.ceil(filteredData.length / pageSize);
      if (currentPage < totalPages) { currentPage++; renderTable(); }
    });

    // CSV Export
    document.getElementById('btn-export-csv').addEventListener('click', () => {
      let csv = '측정시각,경과시간_분,경과시간_시간,온도_원시값,온도_참값,진동_RMS,압력,이벤트\n';
      rawData.forEach((d, idx) => {
        csv += `${timeLabels[idx]},${d.경과시간_분},${d.경과시간_시간},${d.온도_원시값},${d.온도_참값},${d.진동_RMS},${d.압력},"${d.이벤트}"\n`;
      });
      const blob = new Blob([new Uint8Array([0xEF, 0xBB, 0xBF]), csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', 'sensor_smoothed_export.csv');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });

    renderTable();
  }

  // Smooth scroll spy for active navigation item
  function initNavScroll() {
    const navItems = document.querySelectorAll('.nav-item');
    window.addEventListener('scroll', () => {
      let currentSection = '';
      const sections = document.querySelectorAll('.technique-card, .data-table-container');
      sections.forEach(sec => {
        const top = sec.offsetTop - 120;
        if (window.scrollY >= top) {
          currentSection = sec.getAttribute('id');
        }
      });
      navItems.forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('href') === `#${currentSection}`) {
          item.classList.add('active');
        }
      });
    });
  }

  // Run all
  renderKPISummary();
  initSMA();
  initEMA();
  initWMA();
  initSavitzkyGolay();
  initKalman();
  initMedian();
  initZScore();
  initIQR();
  initPeakEnvelopes();
  initBollinger();
  initDecomp();
  initROC();
  initCUSUM();
  initCorrelation();
  initCrossCorrelation();
  initMultiSensor();
  initPhaseSpace();
  initMahalanobis();
  initBenchmark();
  initFFT();
  initDataTable();
  initNavScroll();
});
