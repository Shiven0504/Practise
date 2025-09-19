// Custom Linked List Node class (renamed from Node)
var StudentNode = /** @class */ (function () {
    function StudentNode(data) {
        this.data = data;
        this.next = null;
    }
    return StudentNode;
}());
// Linked List class
var LinkedList = /** @class */ (function () {
    function LinkedList() {
        this.head = null;
    }
    // Insert at specific position (0-based index)
    LinkedList.prototype.insertAt = function (position, data) {
        var newNode = new StudentNode(data);
        if (position < 0) {
            console.log(" Invalid position.");
            return;
        }
        if (position === 0) {
            newNode.next = this.head;
            this.head = newNode;
            return;
        }
        var current = this.head;
        var prev = null;
        var index = 0;
        while (current && index < position) {
            prev = current;
            current = current.next;
            index++;
        }
        if (index !== position) {
            console.log(" Position out of bounds.");
            return;
        }
        if (prev) {
            prev.next = newNode;
            newNode.next = current;
        }
    };
    // Delete node by roll number
    LinkedList.prototype.deleteByRollNo = function (rollNo) {
        if (!this.head) {
            console.log(" List is empty.");
            return;
        }
        if (this.head.data.rollNo === rollNo) {
            this.head = this.head.next;
            return;
        }
        var current = this.head;
        var prev = null;
        while (current && current.data.rollNo !== rollNo) {
            prev = current;
            current = current.next;
        }
        if (!current) {
            console.log(" Roll number not found.");
            return;
        }
        prev.next = current.next;
    };
    // Reverse the linked list
    LinkedList.prototype.reverse = function () {
        var prev = null;
        var current = this.head;
        var next = null;
        while (current) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        this.head = prev;
    };
    // Print the list
    LinkedList.prototype.print = function () {
        var current = this.head;
        if (!current) {
            console.log(" List is empty.");
            return;
        }
        console.log(" Student List:");
        while (current) {
            console.log(" Roll No: ".concat(current.data.rollNo, ", Name: ").concat(current.data.name, ", Grade: ").concat(current.data.grade));
            current = current.next;
        }
    };
    return LinkedList;
}());
// Usage Example
var list = new LinkedList();
list.insertAt(0, { rollNo: 101, name: "Alice", grade: "A" });
list.insertAt(1, { rollNo: 102, name: "Bob", grade: "B" });
list.insertAt(2, { rollNo: 103, name: "Charlie", grade: "C" });
list.print();
console.log("\n Reversing the list...");
list.reverse();
list.print();
console.log("\n Deleting student with Roll No 102...");
list.deleteByRollNo(102);
list.print();
console.log("\n Inserting at position 1...");
list.insertAt(1, { rollNo: 104, name: "Diana", grade: "B+" });
list.print();
