

const fetchWithTimeout = async (resource, options = {}) => {
    const { timeout = 5000 } = options;

    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(resource, {
        ...options,
        signal: controller.signal
    }).finally(() => clearTimeout(id));

    return response;
};


async function fetchAndPlotData(i) {
    const response = await fetchWithTimeout(`/data/${i}`, { timeout: 3000 });
    const dataBatch = await response.json();

    dataBatch.forEach(data => {
        Plotly.newPlot(`cell-${data['i']}-${data['j']}`, [data['plotting_data']], data['layout'], { responsive: true, autosize: true });
    });

};


async function initGrid(rows, cols, xlabel, ylabel) {

    // TODO: add containers for xlabel and ylabel and corresponding text

    const plotContainer = document.getElementById('figure-container');
    const fragment = document.createDocumentFragment();

    for (let i = 0; i < rows; i++) {
        const row = document.createElement('div');
        row.className = 'row';

        for (let j = 0; j < cols; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.id = `cell-${i}-${j}`;
            cell.style.height = `${100 / (rows * 1.2)}vh`;
            cell.style.width = `${100 / cols}vw`;
            row.appendChild(cell);
        }
        fragment.appendChild(row);
    }
    plotContainer.appendChild(fragment);
}


function initSlider(panels) {

    const slider = document.getElementById('slider');
    slider.max = panels - 1;

    const sliderLabel = document.getElementById('slider-label')

    // function to couple slider changes to plot updates
    // client-side throttling to prevent server overloading
    let lastRequestTime = 0;
    const throttleInterval = 100;

    slider.addEventListener('input', async function(event) {
        const value = event.target.value;
        const now = Date.now();
        if (now - lastRequestTime >= throttleInterval) {
            lastRequestTime = now;
            await fetchAndPlotData(value);
            sliderLabel.textContent = `${value}`
        }
    });
    
}


async function init(rows, cols, panels, xlabel, ylabel) {

    initGrid(rows, cols, xlabel, ylabel);

    // // TODO: figure out why this breaks slider behavior!
    // // Fetch and plot initial data
    await fetchAndPlotData(0);
    initSlider(panels);
};