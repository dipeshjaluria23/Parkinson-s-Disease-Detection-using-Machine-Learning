document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const loadingState = document.getElementById('loading-features');
    const autofillBtn = document.getElementById('autofill-btn');
    const modal = document.getElementById('result-modal');
    const closeBtn = document.querySelector('.close-btn');

    // Sample data from the dataset (a healthy person and a parkinson's person)
    // For simplicity, we just use one sample here to show functionality.
    const sampleData = {
        "MDVP:Fo(Hz)": 119.992,
        "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.0037,
        "MDVP:PPQ": 0.00554,
        "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374,
        "MDVP:Shimmer(dB)": 0.426,
        "Shimmer:APQ3": 0.02182,
        "Shimmer:APQ5": 0.0313,
        "MDVP:APQ": 0.02971,
        "Shimmer:DDA": 0.06545,
        "NHR": 0.02211,
        "HNR": 21.033,
        "RPDE": 0.414783,
        "DFA": 0.815285,
        "spread1": -4.813031,
        "spread2": 0.266482,
        "D2": 2.301442,
        "PPE": 0.284654
    };

    // Fetch features to build the form
    fetch('/api/features')
        .then(response => response.json())
        .then(data => {
            loadingState.style.display = 'none';
            if (data.features) {
                data.features.forEach(feature => {
                    const group = document.createElement('div');
                    group.className = 'input-group';
                    
                    const label = document.createElement('label');
                    label.htmlFor = feature;
                    label.textContent = feature;
                    
                    const input = document.createElement('input');
                    input.type = 'number';
                    input.step = 'any';
                    input.id = feature;
                    input.name = feature;
                    input.required = true;
                    
                    group.appendChild(label);
                    group.appendChild(input);
                    form.appendChild(group);
                });
            } else {
                loadingState.innerHTML = '<p class="error">Failed to load features. Ensure backend is running and model is trained.</p>';
            }
        })
        .catch(err => {
            console.error(err);
            loadingState.innerHTML = '<p class="error">Failed to connect to backend.</p>';
        });

    // Handle auto-fill
    autofillBtn.addEventListener('click', () => {
        Object.keys(sampleData).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = sampleData[key];
            }
        });
    });

    // Handle form submission
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const submitBtn = document.querySelector('button[form="prediction-form"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<div class="spinner" style="width:20px;height:20px;border-width:2px;margin:0"></div> Analyzing...';
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(result => {
            showResult(result);
        })
        .catch(err => {
            console.error(err);
            alert("Error making prediction. Please try again.");
        })
        .finally(() => {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        });
    });

    function showResult(result) {
        const icon = document.getElementById('status-icon');
        const text = document.getElementById('result-text');
        const probHealthyFill = document.getElementById('prob-healthy-fill');
        const probHealthyText = document.getElementById('prob-healthy-text');
        const probParkinsonsFill = document.getElementById('prob-parkinsons-fill');
        const probParkinsonsText = document.getElementById('prob-parkinsons-text');

        // Reset widths for animation
        probHealthyFill.style.width = '0%';
        probParkinsonsFill.style.width = '0%';

        if (result.prediction === 1) {
            icon.className = 'status-indicator status-parkinsons';
            icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            text.textContent = "Positive for Parkinson's Disease";
            text.style.color = 'var(--danger)';
        } else {
            icon.className = 'status-indicator status-healthy';
            icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
            text.textContent = "Negative for Parkinson's Disease (Healthy)";
            text.style.color = 'var(--success)';
        }

        modal.classList.add('show');

        // Animate probabilities
        setTimeout(() => {
            const healthyPct = Math.round(result.probability_healthy * 100);
            const parkinsonsPct = Math.round(result.probability_parkinsons * 100);
            
            probHealthyFill.style.width = `${healthyPct}%`;
            probHealthyText.textContent = `${healthyPct}%`;
            
            probParkinsonsFill.style.width = `${parkinsonsPct}%`;
            probParkinsonsText.textContent = `${parkinsonsPct}%`;
        }, 100);
    }

    // Modal Close Logic
    closeBtn.addEventListener('click', () => {
        modal.classList.remove('show');
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    });
});
