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

  // ── 3. 今日交易瀑布圖(動畫長條)──
  var wf = document.getElementById('waterfall');
  if (wf) {
    var trades = [{s:'6770',v:102688},{s:'1802',v:57643},{s:'2344',v:-38963},{s:'3037',v:-37909},{s:'2408',v:-31251}];
    var maxabs = Math.max.apply(null, trades.map(function(t){ return Math.abs(t.v); }));
    wf.innerHTML = trades.map(function(t){
      var w = (Math.abs(t.v) / maxabs * 100).toFixed(1), pos = t.v >= 0;
      return '<div class="wfrow"><span class="wfs">' + t.s + '</span><div class="wfbar"><span class="' +
        (pos ? 'wfg' : 'wfr') + '" data-w="' + w + '"></span></div><span class="wfv ' + (pos ? 'green' : 'red') +
        '">' + (pos ? '+' : '') + t.v.toLocaleString() + '</span></div>';
    }).join('');
    var io2 = new IntersectionObserver(function(es){ es.forEach(function(e){ if (e.isIntersecting){
      wf.querySelectorAll('[data-w]').forEach(function(b, i){ setTimeout(function(){ b.style.width = b.getAttribute('data-w') + '%'; }, i * 130); });
      io2.disconnect();
    }}); }, { threshold: .3 });
    io2.observe(wf);
  }
})();
