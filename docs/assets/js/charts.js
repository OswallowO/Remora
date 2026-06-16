// charts.js — 互動真實數據視覺(lightweight-charts + SVG 動畫),全程式、無截圖
(function(){
  var TV = window.LightweightCharts;

  // ── 1. 互動真實 K 線(今日 6770 / 1802,可縮放拖曳)──
  var kEl = document.getElementById('liveKchart');
  if (kEl && TV) {
    var chart = TV.createChart(kEl, {
      width: kEl.clientWidth, height: kEl.clientHeight,
      layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#8b97a6', fontFamily: 'inherit' },
      grid: { vertLines: { color: 'rgba(42,53,67,.5)' }, horzLines: { color: 'rgba(42,53,67,.5)' } },
      rightPriceScale: { borderColor: '#2a3543' },
      timeScale: { borderColor: '#2a3543', timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
    });
    var candle = chart.addCandlestickSeries({
      upColor: '#16c784', downColor: '#ea3943', borderUpColor: '#16c784',
      borderDownColor: '#ea3943', wickUpColor: '#16c784', wickDownColor: '#ea3943',
    });
    var vol = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' });
    vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    function load(sym) {
      fetch('assets/data/k_' + sym + '.json').then(function(r){ return r.json(); }).then(function(d){
        candle.setData(d.candles); vol.setData(d.volume); candle.setMarkers(d.markers);
        chart.timeScale().fitContent();
      });
    }
    load('6770');
    document.querySelectorAll('.chart-toggle button').forEach(function(b){
      b.addEventListener('click', function(){
        document.querySelectorAll('.chart-toggle button').forEach(function(x){ x.classList.remove('on'); });
        b.classList.add('on'); load(b.getAttribute('data-sym'));
      });
    });
    if (window.ResizeObserver) new ResizeObserver(function(){ chart.applyOptions({ width: kEl.clientWidth, height: kEl.clientHeight }); }).observe(kEl);
  }

  // ── 2. 自我描繪權益曲線(SVG,捲動時畫出來)──
  var eqEl = document.getElementById('equityChart');
  if (eqEl) {
    fetch('assets/data/equity.json').then(function(r){ return r.json(); }).then(function(cum){
      var W = 640, H = 240, pad = 10;
      var max = Math.max.apply(null, cum), min = Math.min(0, Math.min.apply(null, cum));
      var rng = (max - min) || 1;
      var pts = cum.map(function(v, i){
        var x = pad + (W - 2 * pad) * i / (cum.length - 1);
        var y = H - pad - (H - 2 * pad) * (v - min) / rng;
        return [x, y];
      });
      var d = 'M' + pts.map(function(p){ return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' L');
      var y0 = H - pad - (H - 2 * pad) * (0 - min) / rng;
      var area = d + ' L' + pts[pts.length - 1][0].toFixed(1) + ',' + y0.toFixed(1) + ' L' + pts[0][0].toFixed(1) + ',' + y0.toFixed(1) + ' Z';
      eqEl.innerHTML =
        '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" style="width:100%;height:240px;display:block">' +
        '<path class="eqarea" d="' + area + '" fill="rgba(22,199,132,.10)" stroke="none" style="opacity:0"/>' +
        '<path class="eqline" d="' + d + '" fill="none" stroke="#16c784" stroke-width="2" stroke-linejoin="round"/></svg>';
      var line = eqEl.querySelector('.eqline'), areaP = eqEl.querySelector('.eqarea');
      var len = line.getTotalLength();
      line.style.strokeDasharray = len; line.style.strokeDashoffset = len;
      var io = new IntersectionObserver(function(es){ es.forEach(function(e){ if (e.isIntersecting){
        line.style.transition = 'stroke-dashoffset 2.4s ease'; line.style.strokeDashoffset = 0;
        areaP.style.transition = 'opacity 1.2s ease 1.2s'; areaP.style.opacity = 1; io.disconnect();
      }}); }, { threshold: .3 });
      io.observe(eqEl);
    });
  }

  // ── 3. 一年回測回放(變速播放,像 client 變速回放)──
  var rcEl = document.getElementById('replayChart');
  if (rcEl && TV) {
    fetch('assets/data/replay_year.json').then(function(r){ return r.json(); }).then(function(d){
      var pts = d.points;
      var chart = TV.createChart(rcEl, {
        width: rcEl.clientWidth, height: rcEl.clientHeight,
        layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#8b97a6', fontFamily: 'inherit' },
        grid: { vertLines: { color: 'rgba(42,53,67,.4)' }, horzLines: { color: 'rgba(42,53,67,.4)' } },
        rightPriceScale: { borderColor: '#2a3543' }, timeScale: { borderColor: '#2a3543' }, crosshair: { mode: 0 },
      });
      var area = chart.addAreaSeries({ lineColor: '#16c784', topColor: 'rgba(22,199,132,.28)', bottomColor: 'rgba(22,199,132,0)', lineWidth: 2 });
      var base = Date.UTC(2025, 3, 1) / 1000;
      var full = pts.map(function(p, i){ return { time: base + i * 86400, value: p.c }; });
      var dateEl = document.getElementById('rpDate'), pnlEl = document.getElementById('rpPnl'), dayEl = document.getElementById('rpDay');
      var idx = 0, timer = null, speed = 1;
      function render(){
        area.setData(full.slice(0, Math.max(1, idx)));
        var p = pts[Math.min(idx, pts.length) - 1] || pts[0];
        var mm = parseInt(p.d.slice(0, 2), 10), yr = mm >= 4 ? 2025 : 2026;
        dateEl.textContent = yr + '/' + p.d;
        pnlEl.textContent = (p.c >= 0 ? '+' : '') + p.c.toLocaleString();
        pnlEl.className = 'rv mono ' + (p.c >= 0 ? 'green' : 'red');
        dayEl.textContent = Math.min(idx, pts.length) + ' / ' + pts.length;
        chart.timeScale().fitContent();
      }
      function stop(){ if (timer){ clearInterval(timer); timer = null; } document.getElementById('rpPlay').textContent = '▶ 播放'; }
      function play(){
        if (timer){ stop(); return; }
        if (idx >= pts.length) idx = 0;
        document.getElementById('rpPlay').textContent = '⏸ 暫停';
        timer = setInterval(function(){ idx += speed; if (idx >= pts.length){ idx = pts.length; render(); stop(); return; } render(); }, 60);
      }
      document.getElementById('rpPlay').addEventListener('click', play);
      document.getElementById('rpReset').addEventListener('click', function(){ stop(); idx = 0; render(); });
      document.querySelectorAll('.replay-ctrl [data-spd]').forEach(function(b){ b.addEventListener('click', function(){
        document.querySelectorAll('.replay-ctrl [data-spd]').forEach(function(x){ x.classList.remove('on'); });
        b.classList.add('on'); speed = parseInt(b.getAttribute('data-spd'), 10);
      }); });
      idx = 1; render();
      if (window.ResizeObserver) new ResizeObserver(function(){ chart.applyOptions({ width: rcEl.clientWidth, height: rcEl.clientHeight }); }).observe(rcEl);
    });
  }
})();
