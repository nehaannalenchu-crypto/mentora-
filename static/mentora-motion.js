(() => {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelectorAll('[data-motion-toggle]').forEach(button => button.addEventListener('click', () => {
    const canvas = button.closest('.home-canvas');
    const paused = canvas.classList.toggle('motion-paused');
    button.setAttribute('aria-pressed', String(paused));
    button.innerHTML = paused ? '<span aria-hidden="true">▶</span> Play animation' : '<span aria-hidden="true">Ⅱ</span> Pause animation';
  }));
  const cards = document.querySelectorAll('[data-reveal]');
  if (!reduced && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.add('is-visible'); observer.unobserve(entry.target); }
    }), {threshold: 0.08});
    cards.forEach(card => { card.classList.add('will-reveal'); observer.observe(card); });
  }
  let toastTimer;
  document.querySelectorAll('.save-tutor-form').forEach(form => form.addEventListener('submit', async event => {
    event.preventDefault();
    const button = form.querySelector('button');
    if (button.disabled) return;
    button.disabled = true;
    try {
      const response = await fetch(form.action, {method: 'POST', headers: {'Accept': 'application/json'}, credentials: 'same-origin'});
      if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) throw new Error('save failed');
      const result = await response.json();
      button.classList.toggle('is-saved', result.saved);
      button.setAttribute('aria-pressed', String(result.saved));
      button.setAttribute('aria-label', `${result.saved ? 'Unsave' : 'Save'} ${button.dataset.name}`);
      button.querySelector('span').textContent = result.saved ? '♥' : '♡';
      button.classList.remove('heart-pop'); void button.offsetWidth; button.classList.add('heart-pop');
      const toast = document.querySelector('.save-toast');
      if (toast) { toast.textContent = result.saved ? '♥ Added to your saved tutors' : 'Removed from your saved tutors'; toast.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove('show'), 2600); }
    } catch { form.submit(); } finally { button.disabled = false; }
  }));
  const celebration = [...document.querySelectorAll('.flash-success')].some(el => /Booking request sent|Demo payment recorded/.test(el.textContent));
  if (celebration && !reduced) {
    const party = document.createElement('div'); party.className = 'celebration'; party.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 24; i++) { const piece = document.createElement('i'); piece.style.setProperty('--x', `${Math.random()*100}vw`); piece.style.setProperty('--delay', `${Math.random()*.6}s`); piece.style.setProperty('--colour', ['#f26922','#b5a0e6','#8dcbb5','#82bde1'][i%4]); party.appendChild(piece); }
    document.body.appendChild(party); setTimeout(() => party.remove(), 3400);
  }
})();
