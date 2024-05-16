document.addEventListener("DOMContentLoaded", function () {
  const textarea = document.getElementById("emailText");
  const form = document.getElementById("classifyForm");
  const expandableSection = document.getElementById("expandableSection");
  const expandContent = document.getElementById("expandContent");

  function resizeTextarea() {
    textarea.style.height = "auto"; // Reset the height
    textarea.style.height = textarea.scrollHeight + "px"; // Set the height to scroll height
  }

  // Add event listener for input in textarea
  textarea.addEventListener("input", resizeTextarea, false);
  resizeTextarea(); // Initialize the size on page load

  // Toggle expand/collapse for classified text
  expandableSection.addEventListener("click", function () {
    expandableSection.classList.toggle("active");

    if (expandContent.style.maxHeight) {
      expandContent.style.maxHeight = null;
      setTimeout(() => {
        expandContent.style.display = "none";
      }, 300); // Match this duration to the CSS transition duration
    } else {
      expandContent.style.display = "block";
      setTimeout(() => {
        expandContent.style.maxHeight = expandContent.scrollHeight + "px";
      }, 0); // Allow display to take effect before transitioning max-height
    }
  });
});
