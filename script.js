/**
 * Extract every second element from an array based on parity.
 * 
 * Filters array elements by index parity. Returns elements at even indices
 * when start=0, or odd indices when start=1.
 * 
 * @param {number[]} arr - Input array of numbers (or any array type)
 * @param {number} [start=1] - Index parity selector
 *                              - 0: select elements at indices 0, 2, 4, ...
 *                              - 1: select elements at indices 1, 3, 5, ...
 * @returns {number[]} Array of selected elements maintaining original values
 * @throws {TypeError} If arr is not an array or start is not 0 or 1
 * 
 * @example
 * const nums = [1, 2, 3, 4, 5, 6];
 * getEverySecond(nums, 0); // [1, 3, 5]
 * getEverySecond(nums, 1); // [2, 4, 6]
 * getEverySecond(nums);    // [2, 4, 6] (default start=1)
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

// Calculate and display statistics
const sum = result.reduce((acc, val) => acc + val, 0);
const average = sum / result.length;

console.log(`Sum: ${sum}`);
console.log(`Average: ${average}`);

// export for Node.js usage / tests
if (typeof module !== "undefined" && module.exports) {
    module.exports = { getEverySecond };
}
