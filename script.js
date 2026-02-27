/**
 * Return every second element from `arr`.
 * @param {Array} arr - input array (commonly 100 integers)
 * @param {number} [start=1] - parity to start from: 0 => indices 0,2,4...; 1 => indices 1,3,5...
 * @returns {Array} selected elements
 */
function getEverySecond(arr, start = 1) {
    if (!Array.isArray(arr)) throw new TypeError("arr must be an array");
    start = Number(start);
    if (!Number.isInteger(start) || (start !== 0 && start !== 1)) {
        throw new TypeError("start must be 0 or 1");
    }
    return arr.filter((_, i) => (i % 2) === start);
}

// Example: array of integers from 1 to 100
const nums = Array.from({ length: 100 }, (_, i) => i + 1);

// Get every second number (starting from the 2nd element, i.e. indices 1,3,5,...)
const result = getEverySecond(nums);

console.log("Every second number from 1..100 (starting at index 1):");
console.log(result);

// export for Node.js usage / tests
if (typeof module !== "undefined" && module.exports) {
    module.exports = { getEverySecond };
}
