// REMORA — 進階互動動畫(2026-06 大品牌改版):滑鼠跟隨 / 3D 傾斜 / 視差 / 導覽縮小 / 自動版本
(function(){
  var hero = document.querySelector('.hero');
  var nav  = document.querySelector('.nav');
  function onScroll(){
    if(nav) nav.classList.toggle('scrolled', window.scrollY > 24);           // #8 導覽縮小變實心
    if(hero){ var g = hero.querySelector('.hero-glow');                        // #6 視差滾動
      if(g) g.style.transform = 'translate3d(0,' + (window.scrollY * 0.22) + 'px,0)'; }
  }
  window.addEventListener('scroll', onScroll, {passive:true}); onScroll();

  if(hero){                                                                    // #3+#4+聚光燈
    hero.addEventListener('mousemove', function(e){
      var r = hero.getBoundingClientRect(), x = e.clientX-r.left, y = e.clientY-r.top;
      var nx = x/r.width-0.5, ny = y/r.height-0.5;
      hero.style.setProperty('--mx', x+'px'); hero.style.setProperty('--my', y+'px');   // 聚光燈(調淡)
      var grid = hero.querySelector('.hero-grid');                            // #4 網格隨滑鼠位移
      if(grid) grid.style.transform = 'translate3d('+(nx*-16)+'px,'+(ny*-16)+'px,0)';
      var panel = hero.querySelector('.hero-panel');                          // #3 卡片 3D 傾斜
      if(panel) panel.style.transform = 'perspective(1000px) rotateX('+(ny*-6).toFixed(2)+'deg) rotateY('+(nx*8).toFixed(2)+'deg)';
    });
    hero.addEventListener('mouseleave', function(){
      hero.style.setProperty('--mx','50%'); hero.style.setProperty('--my','-12%');
      var p = hero.querySelector('.hero-panel'); if(p) p.style.transform = 'perspective(1000px)';
      var g = hero.querySelector('.hero-grid'); if(g) g.style.transform = '';
    });
  }

  fetch('https://api.github.com/repos/OswallowO/Remora/releases/latest')      // 自動版本號 + 直接下載連結
    .then(function(r){ return r.ok ? r.json() : null; })
    .then(function(d){ if(!d) return;
      if(d.tag_name) document.querySelectorAll('[data-gh-version]').forEach(function(el){ el.textContent = d.tag_name; });
      // 找安裝檔(名稱含 setup 的 .exe;否則任一 .exe)→ 下載鈕直接指向資產,不跳 repo 頁
      var as = d.assets || [];
      var inst = as.filter(function(a){ return /setup/i.test(a.name) && /\.exe$/i.test(a.name); })[0]
              || as.filter(function(a){ return /\.exe$/i.test(a.name); })[0];
      if(inst && inst.browser_download_url){
        document.querySelectorAll('[data-gh-download]').forEach(function(el){
          el.setAttribute('href', inst.browser_download_url);   // GitHub 資產以 attachment 提供 → 直接下載
          el.removeAttribute('target');                          // 同分頁下載,不開新分頁
        });
      }
    })
    .catch(function(){});
})();
