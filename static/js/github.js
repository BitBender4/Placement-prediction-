// Count-up animation for stats
document.querySelectorAll('.stat').forEach(stat => {
    const target = +stat.getAttribute('data-target');
    const valueElem = stat.querySelector('strong');
    let count = 0;
    const increment = target / 100;

    const update = () => {
        count += increment;
        if(count < target){
            valueElem.innerText = Math.ceil(count);
            requestAnimationFrame(update);
        } else {
            valueElem.innerText = target;
        }
    }
    update();
});

// Pie chart for languages
const langCanvas = document.getElementById('langChart');
if(langCanvas){
    const ctx = langCanvas.getContext('2d');

    // Fetch language data from dataset attributes
    const labels = JSON.parse(langCanvas.dataset.labels);
    const values = JSON.parse(langCanvas.dataset.values);

    const langData = {
        labels: labels,
        datasets: [{
            data: values,
            backgroundColor: [
                '#36a2eb', '#ffcd56', '#ff6384', '#4bc0c0', '#9966ff', '#ff9f40'
            ]
        }]
    };

    new Chart(ctx, {
        type: 'pie',
        data: langData,
        options: {
            responsive:true,
            plugins: { legend: { position: 'bottom' } },
            animation: { animateRotate: true, duration: 1500 }
        }
    });
}