// Animate progress bar width smoothly on page load
document.addEventListener("DOMContentLoaded", () => {
  const bar = document.getElementById("progress-bar");
  const score = bar.getAttribute("data-score");
  bar.style.width = "0%";

  setTimeout(() => {
    bar.style.transition = "width 1.5s ease-in-out";
    bar.style.width = score + "%";
  }, 100);
});