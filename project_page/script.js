// No dependencies: progressive enhancement for a static GitHub Pages site.
const dialog = document.querySelector('#figure-dialog');
const expandedImage = document.querySelector('#expanded-figure');
const expandedCaption = document.querySelector('#expanded-caption');
const originalLink = document.querySelector('#original-figure');
const figures = [...document.querySelectorAll('[data-figure]')];
let figureIndex = 0;
let figureTrigger;

function showFigure(index) {
  figureIndex = (index + figures.length) % figures.length;
  const link = figures[figureIndex];
  expandedImage.src = link.href;
  expandedImage.alt = link.dataset.alt;
  expandedCaption.textContent = link.dataset.caption;
  originalLink.href = link.href;
}

figures.forEach((link, index) => {
  link.addEventListener('click', event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || typeof dialog?.showModal !== 'function') return;
    event.preventDefault();
    figureTrigger = link;
    showFigure(index);
    dialog.showModal();
    document.body.style.overflow = 'hidden';
  });
});
document.querySelector('#close-figure')?.addEventListener('click', () => dialog.close());
document.querySelector('#previous-figure')?.addEventListener('click', () => showFigure(figureIndex - 1));
document.querySelector('#next-figure')?.addEventListener('click', () => showFigure(figureIndex + 1));
dialog?.addEventListener('keydown', event => {
  if (event.key === 'ArrowRight') { event.preventDefault(); showFigure(figureIndex + 1); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); showFigure(figureIndex - 1); }
});
dialog?.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const r = dialog.getBoundingClientRect();
  if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
});
dialog?.addEventListener('close', () => {
  document.body.style.overflow = '';
  figureTrigger?.focus();
});

const animation = document.querySelector('#reconstruction-animation');
if (animation) {
  animation.muted = true;
  animation.play().catch(() => {
    document.querySelector('#animation-status').textContent = 'Automatic playback is unavailable. The original animation can be downloaded above.';
  });
  animation.addEventListener('error', () => {
    document.querySelector('#animation-status').textContent = 'Animation could not load. Use Download GIF to view the original animation.';
  });
}
