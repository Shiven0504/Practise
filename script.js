// Simple Tic Tac Toe (browser) — click to play, reset button included
(function () {
    const board = Array(9).fill(null);
    let current = "X";
    let running = true;
    const WIN_COMBOS = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ];

    let statusEl, boardEl, resetBtn;

    function init() {
        // create UI
        const container = document.createElement("div");
        container.style.fontFamily = "Arial, sans-serif";
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.alignItems = "center";
        container.style.gap = "12px";
        container.style.margin = "16px";

        statusEl = document.createElement("div");
        statusEl.textContent = `${current}'s turn`;
        statusEl.style.fontWeight = "600";

        boardEl = document.createElement("div");
        boardEl.style.display = "grid";
        boardEl.style.gridTemplateColumns = "repeat(3, 80px)";
        boardEl.style.gridTemplateRows = "repeat(3, 80px)";
        boardEl.style.gap = "6px";

        for (let i = 0; i < 9; i++) {
            const cell = document.createElement("div");
            cell.dataset.index = String(i);
            cell.style.width = "80px";
            cell.style.height = "80px";
            cell.style.display = "flex";
            cell.style.alignItems = "center";
            cell.style.justifyContent = "center";
            cell.style.fontSize = "32px";
            cell.style.cursor = "pointer";
            cell.style.border = "1px solid #333";
            cell.style.userSelect = "none";
            cell.addEventListener("click", onCellClick);
            boardEl.appendChild(cell);
        }

        resetBtn = document.createElement("button");
        resetBtn.textContent = "Reset";
        resetBtn.style.padding = "6px 12px";
        resetBtn.addEventListener("click", reset);

        container.appendChild(statusEl);
        container.appendChild(boardEl);
        container.appendChild(resetBtn);

        // replace existing body content if desired; otherwise append
        if (document.body.children.length === 0) {
            document.body.appendChild(container);
        } else {
            // append to top for convenience
            document.body.insertBefore(container, document.body.firstChild);
        }

        render();
    }

    function onCellClick(e) {
        if (!running) return;
        const idx = Number(e.currentTarget.dataset.index);
        if (board[idx]) return;
        board[idx] = current;
        if (checkWin(current)) {
            running = false;
            statusEl.textContent = `${current} wins!`;
        } else if (board.every(Boolean)) {
            running = false;
            statusEl.textContent = "Draw";
        } else {
            current = current === "X" ? "O" : "X";
            statusEl.textContent = `${current}'s turn`;
        }
        render();
    }

    function checkWin(player) {
        return WIN_COMBOS.some(combo => combo.every(i => board[i] === player));
    }

    function render() {
        Array.from(boardEl.children).forEach((cell, i) => {
            cell.textContent = board[i] || "";
            cell.style.background = board[i] ? (board[i] === "X" ? "#eef" : "#fee") : "#fff";
        });
    }

    function reset() {
        for (let i = 0; i < 9; i++) board[i] = null;
        current = "X";
        running = true;
        statusEl.textContent = `${current}'s turn`;
        render();
    }

    if (typeof document !== "undefined") {
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", init);
        } else {
            init();
        }
    } else {
        console.warn("Tic Tac Toe script requires a browser DOM to run.");
    }
})();