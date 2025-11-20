// UI Enhancements for Fashion Recommender
(function () {
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  ready(function () {
    // Auto-dismiss flash messages after 4 seconds
    const flashes = document.querySelectorAll('.flash');
    flashes.forEach((el) => {
      setTimeout(() => {
        el.style.transition = 'opacity .3s ease, transform .3s ease';
        el.style.opacity = '0';
        el.style.transform = 'translateY(-6px)';
        setTimeout(() => el.remove(), 300);
      }, 4000);
    });

    // Wardrobe: image preview for file input
    const fileInput = document.querySelector('input[type="file"][name="image_file"]');
    const previewBox = document.querySelector('.image-preview');
    if (fileInput && previewBox) {
      fileInput.addEventListener('change', function () {
        const file = this.files && this.files[0];
        if (!file) {
          previewBox.innerHTML = '<span class="muted">No image</span>';
          return;
        }
        if (!file.type.startsWith('image/')) {
          previewBox.innerHTML = '<span class="muted">Not an image</span>';
          return;
        }
        const reader = new FileReader();
        reader.onload = function (e) {
          previewBox.innerHTML = '';
          const img = document.createElement('img');
          img.src = e.target.result;
          previewBox.appendChild(img);
        };
        reader.readAsDataURL(file);
      });
    }

    // Wardrobe: show hex value for color picker if present
    const colorInput = document.querySelector('input[name="color_hex"][type="color"]');
    const colorReadout = document.querySelector('[data-color-readout]');
    function updateColorReadout() {
      if (colorInput && colorReadout) {
        colorReadout.textContent = colorInput.value || '#000000';
      }
    }
    if (colorInput && colorReadout) {
      colorInput.addEventListener('input', updateColorReadout);
      updateColorReadout();
    }

    // Accessibility: focus styles for keyboard users
    function handleFirstTab(e) {
      if (e.key === 'Tab') {
        document.body.classList.add('user-is-tabbing');
        window.removeEventListener('keydown', handleFirstTab);
      }
    }
    window.addEventListener('keydown', handleFirstTab);

    // Dynamic color swatches (avoid inline CSS in templates)
    const swatches = document.querySelectorAll('.swatch[data-color]');
    swatches.forEach(function (el) {
      const hex = (el.getAttribute('data-color') || '').trim();
      if (hex) {
        el.style.backgroundColor = hex;
      }
    });

    console.log('Fashion Recommender enhancements ready');
  });
})();
