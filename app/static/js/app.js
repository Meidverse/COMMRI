// NeuroScan AI — small progressive enhancements (no dependencies)
(() => {
  const drop = document.getElementById('drop');
  const file = document.getElementById('file');
  if (drop && file) {
    const show = (f) => {
      if (!f) return;
      const url = URL.createObjectURL(f);
      const img = document.getElementById('previewImg');
      img.onload = () => {
        document.getElementById('pmeta').textContent = `${f.name} · ${img.naturalWidth}×${img.naturalHeight}px · ${(f.size / 1024).toFixed(0)} KB`;
      };
      img.src = url;
      drop.classList.add('has');
    };
    file.addEventListener('change', () => show(file.files[0]));
    ['dragover', 'dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(e => drop.addEventListener(e, () => drop.classList.remove('over')));
    drop.addEventListener('drop', ev => {
      ev.preventDefault();
      if (ev.dataTransfer.files.length) { file.files = ev.dataTransfer.files; show(file.files[0]); }
    });
  }
  const form = document.getElementById('analyzeForm');
  if (form) {
    const bp = document.getElementById('bodyPart');
    const sync = () => { const m = form.querySelector('input[name=modality]:checked'); bp.style.display = m && m.value === 'xray' ? 'block' : 'none'; };
    form.querySelectorAll('input[name=modality]').forEach(r => r.addEventListener('change', sync)); sync();
    form.addEventListener('submit', () => document.getElementById('busy').classList.add('on'));
  }
  const blend = document.getElementById('blend');
  if (blend) {
    const v = document.getElementById('viewer');
    const out = document.getElementById('blendVal');
    const upd = () => { v.style.setProperty('--blend', blend.value / 100); out.textContent = blend.value + '%'; };
    blend.addEventListener('input', upd); upd();
  }
  // small screens: sidebar becomes a slide-in drawer
  const menuBtn = document.getElementById('menuBtn');
  if (menuBtn) {
    const setNav = (open) => { document.body.classList.toggle('nav-open', open); menuBtn.setAttribute('aria-expanded', open); };
    menuBtn.addEventListener('click', () => setNav(!document.body.classList.contains('nav-open')));
    document.getElementById('scrim').addEventListener('click', () => setNav(false));
    document.addEventListener('keydown', e => { if (e.key === 'Escape') setNav(false); });
    document.querySelectorAll('#sideNav a').forEach(a => a.addEventListener('click', () => setNav(false)));
  }
  document.querySelectorAll('[data-autosubmit]').forEach(el => el.addEventListener('change', () => el.form.submit()));
})();
