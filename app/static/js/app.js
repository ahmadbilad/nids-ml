const predictionForm = document.getElementById("prediction-form");

if (predictionForm) {
  predictionForm.addEventListener("submit", () => {
    const button = document.getElementById("predict-button");
    const spinner = document.getElementById("predict-spinner");
    const label = document.getElementById("predict-button-label");

    button.disabled = true;
    spinner.classList.remove("d-none");
    label.textContent = "Analyzing...";
  });
}
