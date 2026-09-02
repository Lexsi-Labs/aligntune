/* Make the header title navigate home, matching the logo's behavior. */
document$.subscribe(function () {
  var title = document.querySelector(".md-header__title");
  if (title && !title.dataset.atHome) {
    title.dataset.atHome = "1";
    title.style.cursor = "pointer";
    title.addEventListener("click", function () {
      var logo = document.querySelector(".md-header__button.md-logo");
      if (logo && logo.href) {
        window.location.href = logo.href;
      }
    });
  }
});
