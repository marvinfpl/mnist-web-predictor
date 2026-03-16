function drawOnCanvas() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let draw = false;

    canvas.addEventListener("mousedown", (e) => {
        draw = true;
        ctx.beginPath();
        ctx.lineTo(e.offsetX, e.offsetY);
    });

    canvas.addEventListener("mouseup", () => {
        draw = false;
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!draw) return;

        ctx.lineCap = "round";
        ctx.lineWidth = 5;
        ctx.strokeStyle = "black";

        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    });
}

function clearCanvas() {
    const button = document.getElementById("clear");
    
    button.addEventListener("click", () => {
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");

        ctx.clearRect(0, 0, canvas.width, canvas.height);
    })
}