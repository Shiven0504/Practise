function getEverySecond(arr, start = 1) {
    if (!Array.isArray(arr)) throw new TypeError("arr must be an array");
    start = start === 0 ? 0 : 1; // only allow 0 or 1
    const out = [];
    for (let i = start; i < arr.length; i += 2) out.push(arr[i]);
    return out;
}

// Example: array of integers from 1 to 100
const nums = Array.from({ length: 100 }, (_, i) => i + 1);

// Get every second number (starting from the 2nd element, i.e. indices 1,3,5,...)
const result = getEverySecond(nums);

console.log("Every second number from 1..100 (starting at index 1):");
console.log(result);