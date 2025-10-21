class StudentNode {
    constructor(data) {
        this.data = data;
        this.next = null;
    }
}

class LinkedList {
    constructor() {
        this.head = null;
    }

    insertAt(position, data) {
        if (position < 0) {
            console.error("Invalid position");
            return;
        }

        const newNode = new StudentNode(data);

        if (position === 0) {
            newNode.next = this.head;
            this.head = newNode;
            return;
        }

        let prev = null;
        let current = this.head;
        let index = 0;

        while (current && index < position) {
            prev = current;
            current = current.next;
            index++;
        }

        if (index !== position) {
            console.error("Position out of bounds");
            return;
        }

        prev.next = newNode;
        newNode.next = current;
    }

    deleteByRollNo(rollNo) {
        if (!this.head) return;

        if (this.head.data.rollNo === rollNo) {
            this.head = this.head.next;
            return;
        }

        let prev = null;
        let current = this.head;

        while (current && current.data.rollNo !== rollNo) {
            prev = current;
            current = current.next;
        }

        if (!current) {
            console.warn("Roll number not found");
            return;
        }

        prev.next = current.next;
    }

    reverse() {
        let prev = null;
        let current = this.head;

        while (current) {
            const next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }

        this.head = prev;
    }

    toArray() {
        const out = [];
        let cur = this.head;
        while (cur) {
            out.push(cur.data);
            cur = cur.next;
        }
        return out;
    }

    print() {
        const arr = this.toArray();
        if (arr.length === 0) {
            console.log("List is empty");
            return;
        }
        console.log("Student List:");
        arr.forEach(s => console.log(`Roll No: ${s.rollNo}, Name: ${s.name}, Grade: ${s.grade}`));
    }
}

// Minimal usage example
const list = new LinkedList();
list.insertAt(0, { rollNo: 101, name: "Alice", grade: "A" });
list.insertAt(1, { rollNo: 102, name: "Bob", grade: "B" });
list.insertAt(2, { rollNo: 103, name: "Charlie", grade: "C" });

list.print();
list.reverse();
list.print();

list.deleteByRollNo(102);
list.insertAt(1, { rollNo: 104, name: "Diana", grade: "B+" });
list.print();