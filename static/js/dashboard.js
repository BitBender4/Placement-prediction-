let charts = {};
let statsData = null;
let chartsData = null;

document.addEventListener("DOMContentLoaded", function () {
    initializeDashboard();
});

function initializeDashboard() {
    fetch("/api/dashboard-data")
        .then(response => response.json())
        .then(data => {
            statsData = data.stats;
            chartsData = data.charts;

            loadStats();
            loadCharts();
            setupEventListeners();
        })
        .catch(error => console.error("Error fetching dashboard data:", error));
}

function setupEventListeners() {
    const refreshBtn = document.getElementById("refreshDashboard");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => {
            initializeDashboard(); // just re-fetch everything
        });
    }
}

// === Update Stat Cards ===
function loadStats() {
    if (!statsData) {
        console.error("No statsData found!");
        return;
    }

    document.getElementById("totalPredictions").innerText = statsData.total_predictions || 0;
    document.getElementById("placementRate").innerText = (statsData.placement_rate || 0) + "%";
    document.getElementById("avgSalary").innerText = "₹" + (statsData.avg_salary || 0);
    document.getElementById("modelAccuracy").innerText = (statsData.model_accuracy || 0) + "%";
}

// === Update Charts ===
function loadCharts() {
    if (!chartsData) {
        console.error("No chartsData found!");
        return;
    }

    // Placement Distribution (Pie Chart)
    if (chartsData.placement_distribution) {
        createPlacementDistributionChart(chartsData.placement_distribution);
    }

    // Feature Importance (Bar Chart)
    if (chartsData.feature_importance) {
        createFeatureImportanceChart(chartsData.feature_importance);
    }
}

function createPlacementDistributionChart(distribution) {
    const ctx = document.getElementById("placementDistributionChart").getContext("2d");
    if (charts.placementDistribution) charts.placementDistribution.destroy();

    charts.placementDistribution = new Chart(ctx, {
        type: "pie",
        data: {
            labels: Object.keys(distribution),
            datasets: [{
                data: Object.values(distribution),
                backgroundColor: ["#36A2EB", "#FF6384", "#FFCE56"]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: "bottom" }
            }
        }
    });
}

function createFeatureImportanceChart(importance) {
    const ctx = document.getElementById("featureImportanceChart").getContext("2d");
    if (charts.featureImportance) charts.featureImportance.destroy();

    charts.featureImportance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: importance.labels,
            datasets: [{
                label: "Feature Importance",
                data: importance.values,
                backgroundColor: "rgba(54, 162, 235, 0.6)"
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}