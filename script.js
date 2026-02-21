/* = 20;
var y = 30;
var z = x + y;
console.log("The sum of x and y is: " + z);
*/

// Replace existing content with this script

const nums = Array.from({ length: 100 }, (_, i) => i + 1);
const result = nums.filter(n => (n % 15 !== 0) && (n % 3 === 0 || n % 5 === 0));

console.log("Numbers divisible by 3 or 5 but not 15:");
console.log(result);