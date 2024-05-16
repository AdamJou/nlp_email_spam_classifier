document.addEventListener("DOMContentLoaded", function () {
  const textarea = document.getElementById("emailText");
  const form = document.getElementById("classifyForm");
  const expandableSection = document.getElementById("expandableSection");
  const expandContent = document.getElementById("expandContent");

  function resizeTextarea() {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
  }

  textarea.addEventListener("input", resizeTextarea, false);
  resizeTextarea();

  expandableSection.addEventListener("click", function () {
    expandableSection.classList.toggle("active");

    if (expandContent.style.maxHeight) {
      expandContent.style.maxHeight = null;
      setTimeout(() => {
        expandContent.style.display = "none";
      }, 300);
    } else {
      expandContent.style.display = "block";
      setTimeout(() => {
        expandContent.style.maxHeight = expandContent.scrollHeight + "px";
      }, 0);
    }
  });
});
