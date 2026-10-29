'use strict';
// Keep the entry on the scene selected by the existing UI without changing it.
document.getElementById('realtimeGaussian')?.addEventListener('click', event => {
  event.currentTarget.href = '/fvv' + window.location.search;
});
