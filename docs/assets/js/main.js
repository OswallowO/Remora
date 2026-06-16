// REMORA site — interactions
(function(){
  // Scroll reveal
  var io = new IntersectionObserver(function(es){
    es.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); } });
  }, {threshold:.12});
  document.querySelectorAll('.reveal').forEach(function(el){ io.observe(el); });

  // Mobile nav toggle
  var t = document.querySelector('.nav-toggle'), links = document.querySelector('.nav-links');
  if(t && links){
    t.addEventListener('click', function(){
      links.style.display = (links.style.display==='flex') ? 'none' : 'flex';
      links.style.cssText += ';flex-direction:column;position:absolute;top:62px;right:24px;background:var(--panel);padding:16px 22px;border:1px solid var(--border2);border-radius:10px;gap:14px';
    });
    links.querySelectorAll('a').forEach(function(a){ a.addEventListener('click', function(){
      if(window.innerWidth<=860){ links.style.display='none'; }
    }); });
  }

  // Animate mock candlestick bars (heights from a fixed seq so it looks like a real chart)
  document.querySelectorAll('.mockchart').forEach(function(mc){
    var seq = mc.getAttribute('data-seq');
    var heights = seq ? seq.split(',').map(Number) : [40,55,48,62,58,72,66,80,74,88,82,70,76,90,84];
    heights.forEach(function(h,i){
      var s = document.createElement('span');
      s.style.height = h + '%';
      s.style.background = (i>0 && heights[i]<heights[i-1]) ? 'var(--red)' : 'var(--green)';
      s.style.animationDelay = (i*0.05) + 's';
      mc.appendChild(s);
    });
  });

  // Count-up stats
  document.querySelectorAll('.num[data-to]').forEach(function(el){
    var to = parseFloat(el.getAttribute('data-to')), pre = el.getAttribute('data-pre')||'', suf = el.getAttribute('data-suf')||'';
    var seen=false;
    var o = new IntersectionObserver(function(es){ es.forEach(function(e){ if(e.isIntersecting && !seen){ seen=true;
      var st=null; function step(ts){ if(!st)st=ts; var p=Math.min((ts-st)/1100,1);
        var v=(to*(1-Math.pow(1-p,3))); el.textContent = pre + (Number.isInteger(to)?Math.round(v):v.toFixed(1)) + suf;
        if(p<1)requestAnimationFrame(step); } requestAnimationFrame(step); } }); });
    o.observe(el);
  });

  // Live-ticking chart — simulate a real-time terminal feel (炫泡), no video needed.
  if(!matchMedia('(prefers-reduced-motion:reduce)').matches){
    document.querySelectorAll('.mockchart[data-live]').forEach(function(mc){
      function tick(){
        if(document.hidden) return;
        var bars=mc.querySelectorAll('span'); if(!bars.length) return;
        var hs=[].map.call(bars, function(b){ return parseFloat(b.style.height)||50; });
        hs.shift();
        var last=hs[hs.length-1]||60;
        hs.push(Math.round(Math.max(22, Math.min(96, last + (Math.random()*24-12)))));
        bars.forEach(function(b,i){ b.style.height=hs[i]+'%';
          b.style.background=(i>0 && hs[i]<hs[i-1])?'var(--red)':'var(--green)'; });
      }
      setInterval(tick, 950);
    });
  }
})();
